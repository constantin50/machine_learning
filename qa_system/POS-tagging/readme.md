## Part of Speech Tagging model

Part-of-speech tagging, also called word-category disambiguation, is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech. The model is a convolutional neural network trained to predict POS tags of sentence.
In this case I train it with corpus of tagged questions written in English to analyze structure of a question for entity extraction purposes. 

#### Get Started


#### 1) using default model

```
!git clone https://github.com/constantin50/machine_learning.git
import sys; sys.path.append('./machine_learning')
from pos_tagging import train_model, POSTagger

model = train_model() # train on default questions dataset
tagger = POSTagger(model) # tagger with a default vocabulary and a set of tags 
tagger(["what does a tensor describe","how can i calculate a vector product"])

[['WH', 'AUX', 'DT', 'NOUN', 'VERB'],
 ['WH', 'MOD', 'PRON', 'VERB', 'DT', 'NOUN', 'NOUN']]
```

#### 2) customization 

you can edit model or train it on your own dataset in colab - check pos_tagging.ipynb

```
url = 'your dataset'
data = json.loads(requests.get(url).text)
```

Note, that dataset should be in json format that can be converted into dict type:

```
{'sentence': 'how many years ago did the ship Titanic sink',
 'tags': ['WH', 'ADJ', 'NOUN', 'ADV', 'AUX', 'DT', 'NOUN', 'NOUN', 'VERB']}

 ```


## Dataset of Tagged Questions 

The dataset contains 959 tagged questions on different topics. Originally, the datasrt was created for training pos tagging 
model to analyze queries in QA systems.

```
how do doctors diagnose bone cancer?
is a tensor independent of any basis?
```

To each question a set of POS-tags is associated, the list of tags:

```
'WH' - WH-word (what, where, how, etc.)
'ADV' - adverb (easily, carefully, etc.) 
'MOD' - modal verb (can, should, must, etc.)
'PRON' - pronoun (he, she, it, etc.)
'VERB' - verb (write, open, sleep, etc.)
'TO' - to (must to do)
'DT' - determinator (the, my, her, etc.)
'ADJ' - adjective (pink, small, clever, etc.) 
'NOUN' - noun (digit, curb, tree, etc.)
'PREP' - preposition (of, in, to, etc.) 
'CONJ' - conjunction (or, and, but, etc.)
'NUMB' - number (three, 3, 3.14, etc.)
'PART' - particle (up, down, out, etc.)
'AUX' - auxiliry verb
```

For example:

```
"what is the function of RAM ?" -> ["WH", "AUX", "DT", "NOUN", "PREP", "NOUN"]
"is a result of a dot product scalar ?" -> ["AUX", "DT", "NOUN", "PREP", "DT", "ADJ", "NOUN", "NOUN"]
```

Note that these tags were designed for QA systems to extract entities from questions. That is why they are not so informative
(and sometimes wrong) from linguistics' point of view.


#### Get started 

```
url = 'https://raw.githubusercontent.com/constantin50/machine_learning/master/qa_system/tagger/train_data.json'
data = json.loads(requests.get(url).text)
data[0]

{'sentence': 'what is a transistor', 'tags': ['WH', 'AUX', 'DT', 'NOUN']}
```
