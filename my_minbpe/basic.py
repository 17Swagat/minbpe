from .base import Tokenizer
from .utils import getStats, merge


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        """Here, we perform merging in order to reduce the length of the tokens. (BPE)"""
        textBytes = text.encode("utf-8")
        tokens = list(textBytes)
        initial_len = len(tokens)
        n_merges = vocab_size - 256

        # Vocab(Reserved):
        for i in range(256):
            self.vocab[i] = bytes([i])

        for i in range(n_merges):
            stats = getStats(tokens)
            top_pair = max(stats, key=stats.get)  # pyright: ignore[reportCallIssue, reportArgumentType]
            if stats[top_pair] < 2:
                break
            replace_id = 256 + i
            tokens = merge(tokens, top_pair, replace_id)
            self.merges[top_pair] = replace_id
            self.vocab[replace_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            if verbose:
                print(f"Merge ({i + 1}/{n_merges}): {replace_id} -> {top_pair}")

        final_len = len(tokens)
        print(f"Compression: {initial_len / final_len:.2f}x")

    def encode(self, text: str):
        text_bytes = text.encode("utf-8")
        tokens = list(text_bytes)
        # Compression:
        while True:
            if len(tokens) < 2:
                break
            stats = getStats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = merge(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, ids): 
        byteTxt = b""
        for id in ids:
            tok = self.vocab.get(id)
            if tok is None:
                raise ValueError(f'Token: {id} not part of vocabulary!!')
            
            byteTxt += self.vocab[id]
        txt = byteTxt.decode('utf-8', errors='replace')
        return txt