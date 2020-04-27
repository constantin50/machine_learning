# -*- coding: utf-8 -*-
"""pos_tagging.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y2SQr_OpUCvIy6jKct5XF6NVbc-akNgI
"""

import re
from sklearn.metrics import classification_report
import copy
import traceback
import datetime
import random
import numpy as np
import json
import requests
import collections
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# load dataset
url = 'https://raw.githubusercontent.com/constantin50/machine_learning/master/qa_system/POS-tagging/train_data.json'
data = json.loads(requests.get(url).text)


data = [element for element in data if type(element) is dict]
sents = [element['sentence'] for element in data]
tags = [element['tags'] for element in data]

data[255]

#@title Functions for tokenization and building a vocabulary of tokens

def tokenize_text_simple_regex(txt, min_token_size=4):
	TOKEN_RE = re.compile(r'[\w\d]+')
	txt = txt.lower()
	all_tokens = TOKEN_RE.findall(txt)
	return [token for token in all_tokens if len(token) >= min_token_size]


def char_tokenize(txt):
  return list(txt)


def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, **tokenizer_kwargs):
  return [tokenizer(text, **tokenizer_kwargs) for text in texts]
		

def build_vocabulary(tokenized_texts, max_size=1000000, max_doc_freq=0.8, min_count=5, pad_word=None):
	
	'''
			parameters
			-----------

			tokenized_texts : list of lists of string
				list of tokenized texts

			max_doc_freq : double
				if word frequency is more then a given then it will be removed

			min_count : double
				if word frequency is less then a given then it will be removed

			pad_word : string
				word for padding 

			returns
			---------

			word2id : dict
				numeration of words

			word2freq : numpy array
				frequency of words
	'''
	#count freq of words

	word_counts = collections.defaultdict(int)
	doc_n = 0

	for txt in tokenized_texts:
		doc_n += 1
		unique_text_tokens = set(txt)
		for token in unique_text_tokens:
			word_counts[token] += 1

	# remove too rare and too frequent words
	# the middle of Zipf's law
	#word_counts = {word: cnt for word, cnt in word_counts.items()
	#				if cnt >= min_count and cnt / doc_n <= max_doc_freq}

	# sort by decrise of frequency

	sorted_word_counts = sorted(word_counts.items(),
								reverse = True,
								key = lambda pair: pair[1])

	# add fake token with 0 index

	if pad_word is not None:
		sorted_word_counts = [(pad_word,0)] + sorted_word_counts

	if len(word_counts) > max_size:
		sorted_word_counts = sorted_word_counts[:max_size]

	# numeration of words

	word2id = {word: i for i, (word,_) in enumerate(sorted_word_counts)}

	# weights 
	word2freq = np.array([cnt/doc_n for _, cnt in sorted_word_counts], dtype='float32')

	return word2id, word2freq

#@title Function for convertation datset into tensor form


def pos_corpus_to_tensor(sentences, tags , char2id, label2id, max_sent_len, max_token_len):
    
    """
    parameters
    -------------
    sentences : conll data
    char2id : dict
      dict of enumerated characters
    label2id : dict
      dict of enumerated tags
    max_sent_len : int
    max_token_len : int

    returns
    -----------
    inputs : torch tensor (SizeCorpus x MaxLenSent x MaxLenToken+2)
      tokenized texts
    targets : torch tensor (SizeCorpus x MaxLenSent)
      tags of each sentence in corpus
   
    """

    inputs = torch.zeros((len(sentences), max_sent_len, max_token_len + 2), dtype=torch.long)
    targets = torch.zeros((len(sentences), max_sent_len), dtype=torch.long)

    for i, sent in enumerate(sentences):
        sent = sent.split(" ")
        for j, token in enumerate(sent):
            targets[i, j] = label2id.get(tags[i][j], 0)
            for k, char in enumerate(token):
                inputs[i, j, k + 1] = char2id.get(char, 0)

    return inputs, targets

# build vocabulary of characters, calculate its frequency, numerate them.
train_char_tokenized = tokenize_corpus(sents, tokenizer=char_tokenize)
char_vocab, word_doc_freq = build_vocabulary(train_char_tokenized, max_doc_freq=1.0, min_count=0, pad_word='<PAD>')

# build vocabulary of tags, numerate them.
temp = list()
for sent in tags:
  for tag in sent:
    temp.append(tag)
UNIQUE_TAGS =  ['<NOTAG>'] + list(set(sorted(temp)))
label2id = dict()
for i in range(len(UNIQUE_TAGS)):
  label2id[UNIQUE_TAGS[i]] = i

UNIQUE_TAGS

# convert corpus into tensor and split into train and test datasets
train_x, train_y = pos_corpus_to_tensor(sents, tags, char_vocab, label2id, 40, 20)

train_dataset = TensorDataset(train_x[:900], train_y[:900])
test_dataset = TensorDataset(train_x[900:], train_y[900:])

# train_x[i][j][k] = k_th letter in j_th word in i_th sentence
# train_y[i][j] = tag id of j_th word in i_th sentence

#@title Convolution Block

class StackedConv1d(nn.Module):
  """
  nn.Sequential takes a list of modules and applies them in sequence
  and translate result of previous module in the next one. 


  the first layer is 1d convolution layer
  the second one is dropout for reducing overfitting
  the third - activation function - LeakyReLU
  
  """
  def __init__(self, features_num, layers_n = 1, kernel_size=3, 
               conv_layer = nn.Conv1d, dropout=0.0):
    super().__init__()
    layers = []
    for _ in range(layers_n):
      layers.append(nn.Sequential(
          conv_layer(features_num, features_num, kernel_size, padding=kernel_size//2),
          nn.Dropout(dropout),
          nn.LeakyReLU()))
    self.layers = nn.ModuleList(layers)

  def forward(self, x):
    """ X : tensor (batchSize x featuresNum x seqLen) """
    # ResNet 
    for layer in self.layers:
      x = x + layer(x)
    return x

#@title Taggers

# tagger based on stucture of word only, i.e. context is ignored
class SingleTokenPOSTagger(nn.Module):
  """
  Parameters
  ------------
  vocab_size : int
    number of unique characters
  labels_num : int 
    number of tags
  embedding_size : int
    size of embedding vector

  backbone - ResNet layer
  global_pooling - transform matrix into vector by pooling
  out - applies a linear transformation to the incoming data: x*W.T + b
  """
  def __init__(self, vocab_size, labels_num, embedding_size=32, **kwargs):
    super().__init__()
    self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
    self.backbone = StackedConv1d(embedding_size, **kwargs)
    self.global_pooling = nn.AdaptiveMaxPool1d(1)
    self.out = nn.Linear(embedding_size, labels_num)
    self.labels_num = labels_num
    
  def forward(self, tokens):
    """ tokens : tensor (batchSize x maxSentLen x maxTokenLen)  """

    # reduce 3d tensor to 2d one
    batch_size, max_sent_len, max_token_len = tokens.shape
    tokens_flat = tokens.view(batch_size*max_sent_len, max_token_len)

    # build embeddings
    char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
    char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen

    # send embeddings to backbone to take into account a context of each character 
    features = self.backbone(char_embeddings)

    # thus we have vectors of features for every character but 
    # we want to tag a token, so let us aggregate characters into tokens
    # by pooling. 

    # global pooling take a matrix (NxM) and build vector (N), where i_th element
    # is max element from i_th column of the matrix.

    global_features = self.global_pooling(features).squeeze(-1) # BatchSize*MaxSentLen x EmbSize

    logits_flat = self.out(global_features) # BatchSize*MaxSentLen x LabelsNum
    
    # add sentence's dimension 
    logits = logits_flat.view(batch_size, max_sent_len, self.labels_num)
    logits = logits.permute(0,2,1) # BatchSize x LabelsNum x MaxSentLen
    return logits


# tagger based on context of a word
class SentenceLevelPOSTagger(nn.Module):
  """
  Parameters
  ------------
  vocab_size : int
    number of unique characters
  labels_num : int 
    number of tags
  embedding_size : int
    size of embedding vector

  backbone - ResNet layer
  global_pooling - transform matrix into vector by pooling
  out - applies a linear transformation to the incoming data: x*W.T + b
  """
  def __init__(self, vocab_size, labels_num, embedding_size=32, single_backbone_kwargs={}, context_backbone_kwargs={}):
    super().__init__()
    self.embedding_size = embedding_size
    self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
    self.single_token_backbone = StackedConv1d(embedding_size, **single_backbone_kwargs)
    self.context_backbone = StackedConv1d(embedding_size, **context_backbone_kwargs)
    self.global_pooling = nn.AdaptiveMaxPool1d(1)
    self.out = nn.Conv1d(embedding_size, labels_num, 1)
    self.labels_num = labels_num
    
  def forward(self, tokens):
    """ tokens : tensor (batchSize x maxSentLen x maxTokenLen)  """

    # reduce 3d tensor to 2d one
    batch_size, max_sent_len, max_token_len = tokens.shape
    tokens_flat = tokens.view(batch_size*max_sent_len, max_token_len)

    # build embeddings
    char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
    char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen

    # send embeddings to backbone to take into account a context of each character 
    char_features = self.single_token_backbone(char_embeddings)

    # thus we have vectors of features for every character but 
    # we want to tag a token, so let us aggregate characters into tokens
    # by pooling. 

    # global pooling take a matrix (NxM) and build vector (N), where i_th element
    # is max element from i_th column of the matrix.

    token_features_flat = self.global_pooling(char_features).squeeze(-1) # BatchSize*MaxSentLen x EmbSize
    
    # features of tokens without it's context
    token_features = token_features_flat.view(batch_size, max_sent_len, self.embedding_size)
    token_features = token_features.permute(0,2,1) # batchSize x EmbSize x MaxSentLen
    
    # recalculate features with respect of context
    context_features = self.context_backbone(token_features)

    logits = self.out(context_features) # BatchSize*MaxSentLen x LabelsNum
    return logits


# wrapper for a model
class POSTagger:
    def __init__(self, model, char2id, id2label, max_sent_len, max_token_len):
        self.model = model
        self.char2id = char2id
        self.id2label = id2label
        self.max_sent_len = max_sent_len
        self.max_token_len = max_token_len

    def __call__(self, sentences):
        tokenized_corpus = tokenize_corpus(sentences, min_token_size=0)
        inputs = torch.zeros((len(sentences), self.max_sent_len, self.max_token_len + 2), dtype=torch.long)

        for sent_i, sentence in enumerate(tokenized_corpus):
            for token_i, token in enumerate(sentence):
                for char_i, char in enumerate(token):
                    inputs[sent_i, token_i, char_i + 1] = self.char2id.get(char, 0)

        dataset = TensorDataset(inputs, torch.zeros(len(sentences)))
        predicted_probs = predict_with_model(self.model, dataset)  # SentenceN x TagsN x MaxSentLen
        predicted_classes = predicted_probs.argmax(1)

        result = []
        for sent_i, sent in enumerate(tokenized_corpus):
            result.append([self.id2label[cls] for cls in predicted_classes[sent_i, :len(sent)]])
        return result

#@title Function to make prediction with a given model

def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))

def predict_with_model(model, dataset, device=None, batch_size=32, num_workers=0, return_labels=False):
    """
    Parameters:
    ------------

    model: torch.nn.Module - trained model
    dataset: torch.utils.data.Dataset 
    device: cuda/cpu - device on which computation will done.
    batch_size: int

    Returns:
    --------
    return: numpy.array - len(dataset) x *
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results_by_batch = []

    device = torch.device(device)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = []
    with torch.no_grad():
        import tqdm
        for batch_x, batch_y in tqdm.tqdm(dataloader, total=len(dataset)/batch_size):
            batch_x = copy_data_to_device(batch_x, device)

            if return_labels:
                labels.append(batch_y.numpy())

            batch_pred = model(batch_x)
            results_by_batch.append(batch_pred.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(results_by_batch, 0)

#@title loop for training

def train_eval_loop(model, train_dataset, val_dataset, criterion,
                    lr=1e-4, epoch_n=10, batch_size=32,
                    device=None, early_stopping_patience=10, l2_reg_alpha=0,
                    max_batches_per_epoch_train=10000,
                    max_batches_per_epoch_val=1000,
                    data_loader_ctor=DataLoader,
                    optimizer_ctor=None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n=0):
    """
    Loop for model training.
    
    Parameters
    --------------
    model: torch.nn.Module - model to train
    train_dataset: torch.utils.data.Dataset - train data
    val_dataset: torch.utils.data.Dataset - validation data
    criterion:
    lr: speed of training
    epoch_n: 
    batch_size: 
    device: cuda/cpu
    early_stopping_patience: it is a number, if a number of epochs is higher than it and
    model does not improvment anymore then training is stopped
    l2_reg_alpha: coeff of L2-regularization
    max_batches_per_epoch_train:
    max_batches_per_epoch_val:
    data_loader_ctor: class object for converting dataset into batches
    
    Returns:
    -----------
    return: tuple: (mean value of cost-fuction, the best model)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    if optimizer_ctor is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(model.parameters(), lr=lr)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None

    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                        num_workers=dataloader_workers_n)
    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=dataloader_workers_n)

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()
            print('Epoch {}'.format(epoch_i))

            model.train()
            mean_train_loss = 0
            train_batches_n = 0
            for batch_i, (batch_x, batch_y) in enumerate(train_dataloader):
                if batch_i > max_batches_per_epoch_train:
                    break

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                pred = model(batch_x)
                loss = criterion(pred, batch_y)

                model.zero_grad()
                loss.backward()

                optimizer.step()

                mean_train_loss += float(loss)
                train_batches_n += 1

            mean_train_loss /= train_batches_n
            print('Epoch: {} iterations, {:0.2f} sec'.format(train_batches_n,
                                                           (datetime.datetime.now() - epoch_start).total_seconds()))
            print('mean value of cost function on training dataset', mean_train_loss)



            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):
                    if batch_i > max_batches_per_epoch_val:
                        break

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)

                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)

                    mean_val_loss += float(loss)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print('mean value of cost function on validation dataset', mean_val_loss)

            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                print('model is improved')
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('training stops'.format(
                    early_stopping_patience))
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

            print()
        except KeyboardInterrupt:
            break
        except Exception as ex:
            break

    return best_model

sentence_level_model = SentenceLevelPOSTagger(len(char_vocab), len(label2id), embedding_size=64,
                                              single_backbone_kwargs=dict(layers_n=3, kernel_size=3, dropout=0.3),
                                              context_backbone_kwargs=dict(layers_n=3, kernel_size=3, dropout=0.3))

def train_model(model, train, test):
  model = train_eval_loop(sentence_level_model,
                                              train_dataset,
                                              test_dataset,
                                              F.cross_entropy,
                                              lr=5e-3,
                                              epoch_n=100,
                                              batch_size=40,
                                              device='cuda',
                                              early_stopping_patience=5,
                                              max_batches_per_epoch_train=30,
                                              max_batches_per_epoch_val=20,
                                              lr_scheduler_ctor=lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2,
                                                                                                                         factor=0.5,
                                                                                                                         verbose=False))
  return model

#torch.save(model.state_dict(), '/content/drive/My Drive/new_pos_model.pth')

def load_model(path):
  sentence_level_model.load_state_dict(torch.load(path))
  model = sentence_level_model
  return model

MAX_SENT_LEN = 20
MAX_ORIG_TOKEN_LEN = 20

Tagger = POSTagger(model, char_vocab, UNIQUE_TAGS, MAX_SENT_LEN, MAX_ORIG_TOKEN_LEN)