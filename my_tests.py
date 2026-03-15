"""This file will contains code for testing my `Version of MinBPE`"""

# %%
# My MinBPE Version
from my_minbpe import BasicTokenizer
mytokenizer = BasicTokenizer()

with open('tests/taylorswift.txt', 'r') as file:
    text = file.read()

mytokenizer.train(text, 500, False)
# %%
inputText = 'Awation    '
my_encoding = mytokenizer.encode(inputText)
print(my_encoding)
my_decoding = mytokenizer.decode(my_encoding)
print(my_decoding == inputText)
# %% 
idx_check = 468 #388 #279 #32
print(
    mytokenizer.vocab[idx_check]
)
# %%
inputText = 'write a album'
my_encoding = mytokenizer.encode(inputText)
print(my_encoding)
my_decoding = mytokenizer.decode(my_encoding)
print(my_decoding == inputText)
# %%
for enc in my_encoding:
    print(f'{enc} -> {mytokenizer.vocab[enc]}')

# %%
mytokenizer.vocab

# %%
# @Karpathy:
# Let's Test out `minbpe`:=>

