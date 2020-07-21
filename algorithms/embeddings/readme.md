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
