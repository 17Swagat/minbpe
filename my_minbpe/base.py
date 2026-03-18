import ast

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

    def save(self, file_path:str):
        file_path += '.model'
        with open(file_path, 'w') as f:
            f.write('My MinBPE v-0.1\n')
            f.write(f'${self.merges}\n')
            f.write(f'${self.vocab}\n')
        
        print('Model [Merges] & [Vocab] Saved')
    
    def load(self, model_file_path):
        with open(model_file_path, 'r') as file:
            bpe_version = file.readline().strip()
            # print(bpe_version)
            merges = file.readline().strip()[1:]
            vocab = file.readline().strip()[1:]
        merges = ast.literal_eval(merges)
        vocab = ast.literal_eval(vocab)
        self.merges = merges
        self.vocab = vocab
        print('Saved Tokenizer: [Merges] & [vocab] -> Loaded')
        # return merges, vocab