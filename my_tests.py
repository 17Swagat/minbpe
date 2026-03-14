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
tokenizer.train(text, vocab_size=400, verbose=False)

# %%
# tokenizer.merges
# %%
# tokenizer.vocab

# %% 
# x = {
#     (1, 2): 10,
#     (3, 4): 11,
#     (0, 1): 12,
# }

# y = {
#     (1, 2): 20,
#     (3, 4): 51,
#     (0, 1): 22,
#     (2, 4): 23,
# }
# %%
tokenizer.vocab
# %%
# Encoding:
# print(
#     tokenizer.encode('States of america for ur love')
# )
print(
    len(tokenizer.encode('States of america for ur love'))
)
print(
    len(list('States of america for ur love'.encode('utf-8')))
)

# %%
print(
    tokenizer.encode('h')
)
