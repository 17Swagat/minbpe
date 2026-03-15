"""This file will contains code for testing my `Version of MinBPE`"""

########################################################################
# %% Revision #2
from my_minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
with open('tests/taylorswift.txt', 'r') as file:
    text = file.read()

tokenizer.train(text, 400, False)
# %%
# Encoding & Decoding
tempText = "You love Taylor Swift"
encoding = tokenizer.encode(tempText)
print(list(tempText.encode('utf-8')))
print(encoding)

# %%
decoding = tokenizer.decode(encoding)
print(decoding)
print(decoding == tempText)
# %%
tokenizer.decode([128])


# %%
# tokenizer.merges
tokenizer.vocab


# %% Revision #2
# from my_minbpe import getStats, merge

# x = "aaabdaaabac"
# stats = getStats(x)
# top_pair = max(stats, key=stats.get)
# print(top_pair)
# y = merge(x, top_pair, 'P')
# print(list(x))
# print(y)
# # %%
# stats = getStats(y)
# top_pair = max(stats, key=stats.get)
# print(top_pair)
# z = merge(y, top_pair, 'Q')
# print(list(y))
# print(z)
# # %%
# stats = getStats(z)
# top_pair = max(stats, key=stats.get)
# print(top_pair)
# a = merge(z, top_pair, 'M')
# print(list(z))
# print(a)
# # %%
# stats = getStats(a)
# top_pair = max(stats, key=stats.get)
# print(top_pair)
# print(stats[top_pair])
# b = merge(a, top_pair, 'N')
# print(list(a))
# print(b)

# %%
########################################################################

# # %%
# # My MinBPE Version
# from my_minbpe import BasicTokenizer
# mytokenizer = BasicTokenizer()

# with open('tests/taylorswift.txt', 'r') as file:
#     text = file.read()

# mytokenizer.train(text, 500, False)
# # %%
# inputText = 'Awation    '
# my_encoding = mytokenizer.encode(inputText)
# print(my_encoding)
# my_decoding = mytokenizer.decode(my_encoding)
# print(my_decoding == inputText)
# # %% 
# idx_check = 468 #388 #279 #32
# print(
#     mytokenizer.vocab[idx_check]
# )
# # %%
# inputText = 'write a album'
# my_encoding = mytokenizer.encode(inputText)
# print(my_encoding)
# my_decoding = mytokenizer.decode(my_encoding)
# print(my_decoding == inputText)
# # %%
# for enc in my_encoding:
#     print(f'{enc} -> {mytokenizer.vocab[enc]}')

# # %%
# mytokenizer.vocab




# # %%
# # @Karpathy:
# # Let's Test out `minbpe`:=>

