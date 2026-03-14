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
        if verbose:
            print(f"Compression: {initial_len / final_len:.2f}x")

        self.merges = merges
        
        # Developing the Vocab:
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), id in self.merges.items():
            self.vocab[id] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text): ...

    def decode(self, ids): ...
