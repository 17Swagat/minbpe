import regex as re
from tqdm import tqdm

from .base import Tokenizer
from .utils import getStats, merge


class RegexTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.gpt4Pattern = re.compile(self.GPT4_SPLIT_PATTERN)
        # self.merges = {}  # e.g: {(97, 108): 274, ...}
        # self.vocab = {}   # e.g: {1: b'\xx', ....}

    def train(self, text: str, vocab_size, verbose=False):
        assert vocab_size > 255, "vocab_size needs to be greater than 255"
        self.vocab = {i: bytes([i]) for i in range(256)}

        ids = re.findall(
            self.gpt4Pattern, text
        )  # ['txt1', 'txt2', ...] # "Helping in splitting text"
        ids = [list(chunk.encode("utf-8")) for chunk in ids]

        n_merges = vocab_size - 255
        for i in tqdm(range(n_merges)):
            stats = {}
            for chuck_id in ids:
                getStats(chuck_id, stats)
            top_pair = max(stats, key=stats.get)  # pyright: ignore[reportCallIssue, reportArgumentType]
            replace_id = 256 + i
            ids = [merge(chunk_id, top_pair, replace_id) for chunk_id in ids]
            self.merges[top_pair] = replace_id
            self.vocab[replace_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
    def encode(self, text: str): 
        tokens = list(text.encode('utf-8'))
        stats = {}
        while True:
            if (len(tokens) < 2):
                break
            stats = getStats(tokens)
            pair = min(stats,key= lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            replace_id = self.merges[pair]
            tokens = merge(tokens, pair, replace_id)
        
        return tokens
                
    def decode(self, ids): 
        txt_bytes = b""
        for id in ids:
            if id not in self.vocab:
                raise ValueError(f'ID: {id} NOT in Vocab')
            txt_bytes += self.vocab[id]
        txt = txt_bytes.decode('utf-8', errors='replace')
        return txt
    
    # def encode(self, text: str): 
    #     tokens = list(text.encode('utf-8'))
    #     i = 0
    #     ids = []
    #     while True:
    #         if len(tokens) < 2:
    #             break
    #         stats = getStats(tokens)
    #         pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
    #         if pair not in self.merges:
    #             break
    #         tokens = merge(tokens, pair, self.merges[pair])
    #     return tokens
        
        # while i < len(tokens):
        #     if i == len(tokens) - 1:
        #         ids += [tokens[i]]
        #         break
        #     pair = (tokens[i], tokens[i+1])
        #     if pair in self.merges:
        #         ids += [self.merges[pair]]
        #         i += 2
        #     else:
        #         ids += [tokens[i]]
        #         i += 1
        # return ids

