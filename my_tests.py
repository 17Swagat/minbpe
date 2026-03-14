"""This file will contains code for testing my `Version of MinBPE`"""

# %%
# Let's Test out `minbpe`:=>

# %%
# My MinBPE Version
from my_minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
  
# %%
# tokenizer.train("aaabdaaabac", vocab_size=300, verbose=True)
# %%
# print(tokenizer.merges)
# %%
# Training on the Taylor Swift.txt
with open('./tests/taylorswift.txt', 'r') as f:
    text = f.read()
print(len(text))
print(len(list(text.encode('utf-8'))))

# %%
tokenizer.train(text, vocab_size=306, verbose=True)

# %%
tokenizer.merges
# %%
tokenizer.vocab
