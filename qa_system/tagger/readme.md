## Dataset 

Dataset contains 450 tagged questions on different topics

```
how do doctors diagnose bone cancer?
is a tensor independent of any basis?
```

to each question set of POS-tags is assigned, list of tags:

'WH' - WH-word (what, where, how, etc.)
'ADVB' - adverb (easily, carefully, etc.) 
'MOD' - modal verb (can, should, must, etc.)
'PRON' - pronoun (he, she, it, etc.)
'VERB' - verb (write, open, sleep, etc.)
'TO' - to as part of infinitive (to compute)
'DT' - determinator (the, my, her, etc.)
'ADJ' - adjective (pink, small, clever, etc.) 
'NOUN' - noun (digit, curb, tree, etc.)
'PREP' - preposition (of, in, to, etc.) 
'CONJ' - conjunction (or, and, but, etc.)
'NUMB' - number (three, 3, 3.14, etc.)
'RP' - particle (up, down, out, etc.)
'AUX' - auxiliry verb

For example:

"what is the function of RAM ?" -> ["WH", "AUX", "DT", "NOUN", "PREP", "NOUN"]
"is a result of a dot product scalar ?" -> ["AUX", "DT", "NOUN", "PREP", "DT", "ADJ", "NOUN", "NOUN"]

Note that these tags were designed for QA systems to extract entities from questions. That is why they are not so informative
(and sometimes wrong) from linguistics' point of view.
