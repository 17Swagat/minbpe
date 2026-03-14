# Following Technical Blog:
# https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers.html#solveit

# %%
# import torch
text = 'This is some text dataset hello, and hi some words!'
# get the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# %% 
print('Hello')

# %%
# Get Unicode code point for English character
print(f"ord('h') = {ord('h')}")

# Get Unicode code point for emoji
print(f"ord('🤗') = {ord('🤗')}")

# Get Unicode code point for Korean character
print(f"ord('안') = {ord('안')}")

# %%
txt = "😑🚀"
bytes_ = txt.encode('utf-8')
tokens = list(bytes_)
print(tokens) # [240, 159, 145, 145, 240, 159, 154, 128]

