class Tokenizer:
    def __init__(self):
        self.merges = {}  # e.g: {(97, 108): 274, ...}
        self.vocab = {}   # e.g: {1: b'\xx', ....}

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError
