import re
import collections
import numpy as np


# functions of tokenization
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
	#count frequency of words
	word_counts = collections.defaultdict(int)
	doc_n = 0

	for txt in tokenized_texts:
		doc_n += 1
		unique_text_tokens = set(txt)
		for token in unique_text_tokens:
			word_counts[token] += 1

	# remove too rare and too frequent words
	# the middle of Zipf's law
	# word_counts = {word: cnt for word, cnt in word_counts.items()
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



# transforms a corpus into a tensor 
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
