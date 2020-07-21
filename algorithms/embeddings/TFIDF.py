# Constantin Werner | const.werner@gmail.com 
# 21.07.2020



import torch
import math

def TF_IDF(w, d, D):
  tf = d.count(w) / len(d)

  total_count = 0 # number of document where the word occurs

  for d_i in D:
    for w_i in d_i:
      if (w == w_i): 
        total_count += 1

  if (total_count != 0): 
    idf = math.log(len(D)/total_count)
  else:
    idf = math.log(len(D))
  return tf*idf


def tfidf_embbedings(D):

  # returns tensor with embeddings for each document

  DIM = max([len(d) for d in D]) # dim of embeddings

  result = torch.zeros((len(D), DIM), dtype=torch.float32)

  for i in range(len(D)):
    for j in range(DIM):
      if j < len(D[i]):
        result[i][j] = TF_IDF(D[i][j], D[i], D)
  
  return torch.tensor(result)


D = [["a","dog","likes","a", "bone"], ["a", "cat","likes", "a", "fish"], ["my", "truck", "is", "red"]]

embs = tfidf_embbedings(D)

print(embs)