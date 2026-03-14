from .base import Tokenizer
from .utils import getStats, merge


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        bytes_ = text.encode("utf-8")
        tokens = list(bytes_)
        initial_len = len(tokens)
        n_merges = vocab_size - 256
        merges = {}
        for i in range(n_merges):
            stats = getStats(tokens)  # {pair: count, ...}
            top_pair = max(stats, key=stats.get)  # pyright: ignore[reportCallIssue, reportArgumentType]
            if stats[top_pair] == 1:
                break
            replace_id = 256 + i
            tokens = merge(tokens, top_pair, replace_id)
            merges[top_pair] = replace_id
            if verbose:
                print(f"Merge {i}/{n_merges}: {top_pair} -> {replace_id}")

        final_len = len(tokens)
        # if verbose:
        print(f"Compression: {initial_len / final_len:.2f}x")

        self.merges = merges

        # Developing the Vocab:
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), id in self.merges.items():
            self.vocab[id] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while True:
            stats = getStats(ids)
            lookfor_pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if lookfor_pair not in self.merges:
                break
            replace_id = self.merges[lookfor_pair]
            ids = merge(ids, lookfor_pair, replace_id)
        
        return ids

    def decode(self, ids):
        # raw_bytes = b"".join([self.vocab[ids] for id in ids])
        raw_bytes = b""
        for id in ids:
            raw_bytes += self.vocab[id]
        txt = raw_bytes.decode('utf-8', errors='replace')
        return txt
            # if id < 256:
            #     append_ = self.vocab[id]
            
            # raw_bytes += append_  # pyright: ignore[reportPossiblyUnboundVariable]
