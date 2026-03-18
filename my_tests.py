"""This file will contains code for testing my `Version of MinBPE`"""

# %%
from my_minbpe import RegexTokenizer
import os

tokenizer = RegexTokenizer()
model_file_path = 'my_models'
model_file_path = os.path.join(model_file_path, 'models_2_400vocab.model')
tokenizer.load(model_file_path)

# %%
# txt = "hello world!!!? (안녕하세요!) lol123 😉"
txt = "안"
txt_encoding = tokenizer.encode(txt)
print(txt_encoding)
txt_decoding = tokenizer.decode(txt_encoding)
print(txt_decoding)

# %%
tokenizer.vocab

# %%

######################################################################################
######################################################################################


# %%
import os
model_file_path = 'my_models'
model_file_path = os.path.join(model_file_path, 'models_2_400vocab.model')
with open(model_file_path, 'r') as file:
    bpe_version = file.readline().strip()
    merges = file.readline().strip()[1:]
    vocab = file.readline().strip()[1:]
print(vocab)
# print(bpe_version)
# print(merges)

# %%
from my_minbpe import RegexTokenizer
import os

tokenizer = RegexTokenizer()
model_file_path = 'my_models'
model_file_path = os.path.join(model_file_path, 'models_2_400vocab.model')
merges, vocab = tokenizer.load(model_file_path)

# %%
# print(merges)
# print(vocab)
print(type(vocab))


# %%
########################################################################
# %%
import os
# import json
import ast

model_file_path = 'my_models'
model_file_path = os.path.join(model_file_path, 'models.model')
with open(model_file_path, 'r') as file:
    bpe_version = file.readline().strip()
    # print(bpe_version)
    merges = file.readline().strip()[1:]
    # merges = "\"" + merges + "\""
# merges = json.loads(merges)
merges = ast.literal_eval(merges)
print(type(merges))
print(merges[(101, 114)])


# %%
########################################################################
# %%
from my_minbpe import RegexTokenizer

tokenizer = RegexTokenizer()
with open('tests/taylorswift.txt', 'r') as file:
    text = file.read()

# tokenizer.train(text, 276, False)
tokenizer.train(text, 400, False)

# %%
# Saving merges
import os
MODEL_SAVE_PATH = 'my_models'
os.makedirs(MODEL_SAVE_PATH,exist_ok=True)
model_file_name = 'models_2_400vocab'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, model_file_name)
tokenizer.save(MODEL_SAVE_PATH)

# %%
########################################################################
# %%
from my_minbpe import RegexTokenizer

tokenizer = RegexTokenizer()
with open('tests/taylorswift.txt', 'r') as file:
    text = file.read()
# %%
tokenizer.train(text, vocab_size=400)
# %%
txt = "\n"
txt_encoding = tokenizer.encode(txt)
print(txt_encoding)
# %%
txt_decoding = tokenizer.decode(txt_encoding)
print(txt_decoding)

# %%
txt = " The her" #op her
txt_encoding = tokenizer.encode(txt)
print(txt_encoding)
# %%
# txt_encoding = [97]
txt_decoding = tokenizer.decode(txt_encoding)
print(txt_decoding)
# %%
txt_encoding = [400]
txt_decoding = tokenizer.decode(txt_encoding)
print(txt_decoding)
# %%
txt = "!"
txt_encoding = tokenizer.encode(txt)
print(txt_encoding)
txt_decoding = tokenizer.decode(txt_encoding)
print(txt_decoding)

# %%
# tokenizer.merges

# %%
tokenizer.vocab

# %%

########################################################################
# %% Implementing `RegexTokenizer`
import regex as re
from my_minbpe import getStats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
gpt4Pattern = re.compile(GPT4_SPLIT_PATTERN)
text_chunks = re.findall(gpt4Pattern, "hello world!")

with open('tests/taylorswift.txt', 'r') as file:
    text = file.read()

# %%
text_chunks = re.findall(gpt4Pattern, text)

# %%
ids = [list(chunk.encode('utf-8')) for chunk in text_chunks]
print(ids[:5])
# %%
n_vocab = 260
n_merges = n_vocab - 256
for i in range(n_merges):
    stats = {}
    for chunk_id in ids:
        getStats(chunk_id, stats)
    top_pair = max(stats, key=stats.get)
    replace_id = 256 + i
    ids = [merge(chuck_id, top_pair, replace_id) for chuck_id in ids]
    

# %%
########################################################################
# %% Revision #2
# from my_minbpe import BasicTokenizer

# tokenizer = BasicTokenizer()
# with open('tests/taylorswift.txt', 'r') as file:
#     text = file.read()

# tokenizer.train(text, 400, False)
# # %%
# # Encoding & Decoding
# tempText = "You love Taylor Swift"
# encoding = tokenizer.encode(tempText)
# print(list(tempText.encode('utf-8')))
# print(encoding)
# # %%
# from minbpe import BasicTokenizer

# karpathyTokenizer = BasicTokenizer()
# karpathyTokenizer.train(text, 400, False)

# # %%
# tempText = "You love Taylor Swift"
# encoding = karpathyTokenizer.encode(tempText)
# print(list(tempText.encode('utf-8')))
# print(encoding)


# # %%
# decoding = tokenizer.decode(encoding)
# print(decoding)
# print(decoding == tempText)
# # %%
# tokenizer.decode([128])


# # %%
# # tokenizer.merges
# tokenizer.vocab


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
