import re
import json
import numpy as np
import collections
file = open(r'C:\projects\ML\dataset.json', 'r', encoding='utf-8')
data = file.read()
data_dict = [json.loads(line) for line in open(r'C:\projects\ML\dataset.json', 'r')]

for i in range(int((len(data_dict)+1)/4),int((len(data_dict)+1)/2)):
	x_test.append(data_dict[i]['short_description'])
	y_test.append(data_dict[i]['category'])

def tokenize_corpus(corpus):
	for i in range(len(corpus)):
		corpus[i] = tokenizer(corpus[i])
	return corpus

def tokenizer(string):
	string = string.lower()
	TOKEN_RE = re.compile(r'[\w\d]+')
	tokens = TOKEN_RE.findall(string)
	return [token for token in tokens]

def build_vocabulary(tokenized_texts, max_size=1000000, max_doc_freq=0.8, min_count=5, pad_word=None):

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

	word_counts = {word: cnt for word, cnt in word_counts.items()
					if cnt >= min_count and cnt / doc_n <= max_doc_freq}

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

def vectorize_texts(tokenized_texts, word2id, word2freq, 
	mode = 'tfidf', scale=True):
	
	# rectangle matrix n*m, where n - num of texts
	# m - num of unique tokens

	result = scipy.sparse.dok_matrix((len(tokenized_texts),
									len(word2id)), dtype='float32')

	for text_i, text in enumerate(tokenized_texts):
		for token in text:
			if token in word2id:
				result[text_i, word2id[token]] += 1

	# algorithms of weight calculation

	# binary vector

	if (mode == 'bin'):
		result = (result > 0).astype('float32')

	# term frequency
	# frequency of a term / length of the document
	elif (mode == 'tf'):
		result = result.tocsr()
		result = result.multiply(1/result.sum(1))


	# inverse document frequency
	# 
	elif (mode == 'itf'):
		result = (result>0).astype('float32').multiply(1/word2freq)

	# tf-itf

	elif (mode == 'tfidf'):
		result = result.tocsr()
		result = result.multiply(1/result.sum(1))
		result = result.multiply(1/word2freq)

	# squeze elements of matrix into interval [0,1]
	# min max standartization
	if scale:
		result = result.tocsc()
		result -= result.min()
		result /= (result.max() + 1e-6)
	
	return result.tocsr()	