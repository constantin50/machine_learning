# Closed domain Q&A system

## Overview

Question answering system is a systems that automatically answer questions posed by humans in a natural language. Closed-domain 
question answering deals with questions under a specific domain. The system of this project handles domain of mathematics. 

## Architecture

The system consists of the follwing parts: 

1. language model BERT traind on Stanford Question Answering Dataset. It gives predictions on where an answer to the question
begins and ends in the given context.
2. Wikipedia API.
3. Analyzer. It handles text data: lemmatization, tagging and correction of spelling.
4. Bot. It extracts questions and entities from user's query. 

![diagram](https://github.com/constantin50/machine_learning/blob/master/qa_system/diagram.png)


# Installing

Prerequisites: the following libraries should be installed: deeppavlov, tensorflow, wikipeadiaapi, nltk, spellchecker 

## Google Colab
just run qa_system.ipynb

## Local host




# Evaluation 

In order to evaluate accuracy of the model the following test was done. The list of 50 pairs [question,keyword] 
was made. Questions in list concern:

1.defintion (e.g. what is an unit vector?);
2.proprities (e.g. is the real numbers uncountable?);
3.interpretations (e.g. what is a determinant geometrically speaking?);
4.examples (e.g. what is a example of a field?).

A keyword to the question is a word that must be presented in answer to this question. For example, here is the question 
"what is an unit vector?" and here is definition of unit vector: "a unit vector in a normed vector space is a vector of length 1".
Keyword is the word "length" for it is an essetinal proprity of a unit vector. So, if this word is missed in the answer then
it is an error. 

# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


