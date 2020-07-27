Word embedding is the collective name for a set of language modeling and feature learning techniques 
in natural language processing where words or phrases from the vocabulary are mapped to vectors of real numbers.

There several approches to it

### Cosine similarity

### One Hot Encoding 

The One Hot Encoding is the simplest approach. Each word takes a vector with size of n, where n is a number 
of unique words. This vector is fill with zeroes but in kth position there is '1' and k is a number of this word
in vocabulary (which is just a map from words to numbers). The sum of such vectors for each word in document represents
a vector for this document.

<img src="https://sun4-17.userapi.com/c857728/v857728972/21419c/N4Wp8caAAjc.jpg" width="450" height="300">

### TF-IDF

TFIDF (short for term frequency–inverse document frequency) is a statistic that reflects how important a word is to a document in a collection.

Firstly, we count how many times a word occurs in a document and devide it by a number of occuring of this word in whole collection

![form1](https://sun9-28.userapi.com/_dFz76KVkloQNW80rxE_b6I61CEGtMshCJmznw/flDfkiUcaXs.jpg)

Secondly, we count how many documents we have and divide it by a number of document where the word occurs.

![form2](https://sun9-70.userapi.com/82kJ_eHhdLmxxDXz6gqIGS4BMPodlMhdDSiBsw/hlTS5ogIldk.jpg)

Then, multiply it.

![form3](https://sun9-74.userapi.com/4rEemv0n1mscw-Ed9nqD6qHU2LVSXIkgcNppdw/pz2sIFVHBfQ.jpg)

### Word2Vec

Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct contexts of words. 

Skip-Gram architecture

In the Skip-Gram model, the goal is to predict a context with the given word. The core idea behind this is distributional hypothesis in linguistics that is often
discribed as the following: "a word is characterized by the company it keeps" 

```

A language is a structured system of communication. Language, in a broader sense, is the method of communication that involves the use of – particularly human – languages. The scientific study of language is called linguistics. Questions concerning the philosophy of language, such as whether words can represent experience, have been debated at least since Gorgias and Plato in ancient Greece. Thinkers such as Rousseau have argued that language originated from emotions while others like Kant have held that it originated from rational and logical thought. Twentieth century philosophers such as Wittgenstein argued that philosophy is really the study of language. Major figures in linguistics include Ferdinand de Saussure and Noam Chomsky.

...

similarity(new_model.W1.weight[word2idx["chomsky"]], new_model.W1.weight[word2idx["saussure"]])

tensor(0.5768)

```

We know that Saussure and Chomsky are linguists and in this sense their names are close semanticly. But for distributional semantics theory, important is that these
names appear in same context. 
