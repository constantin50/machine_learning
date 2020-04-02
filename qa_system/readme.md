# Context-based Closed Domain Q&A System

## Overview

Question answering system is a system that automatically answers questions posed by humans in a natural language. Closed-domain question answering deals with questions under a specific domain. The system of this project handles the domain of mathematics. Context-based means that the system is only able to extract an answer from some context related to the entity presented in the question. 

## Architecture

The system consists of the follwing parts: 

1. language model BERT traind on Stanford Question Answering Dataset. It gives predictions on where an answer to the question
begins and ends in the given context.

2. Wikipedia API.

3. Analyzer. It handles text data: lemmatization, tagging and correction of spelling.

4. Bot. It extracts questions and entities from user's query. 

![diagram](https://github.com/constantin50/machine_learning/blob/master/qa_system/diagram.png)


## Get started


"""
bot = Bot()
bot.take_query()

query: what is a binary relation?
QUESTION: what is binary relation ?
ANSWER: a set of ordered pairs
"""

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

So, accuracy of the model is <b>66%</b>

Questions that concern definitions are handled quite easily since every Wikipedia page starts with definition. Whereas examples
often are not presented, thus, the system often is not able to find an answer to questions about examples. Moreover, context-based
systems are very sensitive to words, so, synonyms are a problem.  


