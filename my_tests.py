"""This file will contains code for testing my `Version of MinBPE`"""

# %%
# Let's Test out `minbpe`:=>

# %%
# My MinBPE Version
from my_minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
  
# % %
# tokenizer.train("aaabdaaabac", vocab_size=300, verbose=True)
# print(tokenizer.merges)
# Training on the Taylor Swift.txt
with open('./tests/taylorswift.txt', 'r') as f:
    text = f.read()
print(len(text))
print(len(list(text.encode('utf-8'))))

# % %
tokenizer.train(text, vocab_size=500, verbose=False)

# % %
txt = "Sept album September"
my_encoding = tokenizer.encode(txt)
my_decoded_text = tokenizer.decode(my_encoding)
print(my_encoding)
print(my_decoded_text)
print(txt == my_decoded_text)

# %%
    
# %%
print(tokenizer.decode([469, 470, 472]))
    
# %%
tokenizer.vocab


# %%
print('hello')
# %%
