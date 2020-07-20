# Constantin Werner | const.werner@gmai.com 
# 20.07.2020

import torch

def make_word2idx(D):
  i = 0
  result = dict()
  for d in D:
    for w in d.split(' '):
      if w not in result:
        result[w] = i
        i += 1
  return result

# encoding word in the frame of document where it occurs
def OHEword(word, doc, word2idx):
  emb_word = torch.zeros((len(word2idx)))
  for word_i in doc.split(' '):
    if (word_i == word):
      emb_word[word2idx[word]] += 1
  return emb_word


# encoding document in corpus
def OHEdoc(doc, word2idx):
  emb_text = torch.zeros((len(word2idx)))
  for word in doc.split(" "):
    emb_text[word2idx[word]] += 1
  
  return emb_text


def similarity(v,u):
  return torch.dot(v,u)/(torch.norm(v)*torch.norm(u))


D = ["a dog likes a bone", "a cat likes a fish"]

word2idx = make_word2idx(D)
v0 = OHEword("dog", D[0],word2idx)
v1 = OHEword("cat", D[1], word2idx)

d0 = OHEdoc(D[0], word2idx)
d1 = OHEdoc(D[1], word2idx)

similarity(d0,d1)