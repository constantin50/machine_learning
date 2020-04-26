## Dataset of Tagged Questions 

The dataset contains 772 tagged questions on different topics. Originally, the datasrt was created for training pos tagging 
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


## Get started 

```
url = 'https://raw.githubusercontent.com/constantin50/machine_learning/master/qa_system/tagger/train_data.json'
data = json.loads(requests.get(url).text)
data[0]

{'sentence': 'what is a transistor', 'tags': ['WH', 'AUX', 'DT', 'NOUN']}
```
