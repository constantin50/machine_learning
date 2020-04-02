import re
import nltk
import wikipediaapi
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# this class handles text data: lemmatization, tagging and correction of spelling.
class Analyzer:

  """
  Attributes
  -----------
  lemmatizer : class from nltk lib
  	class turns word in lemma
  
  spell : class fron spellchecker lib
    class corrects misspelling 
  """

  def __init__(self):
    self.lemmatizer = WordNetLemmatizer()
    self.spell = SpellChecker()


  def normalize(self, exp):
    """
    Parameters
    -----------
    exp : string
  	   expression to normalize

    Returns
    --------
    result : list
  	    list of lemmatized and correctly spelled words

    """
    exp = nltk.word_tokenize(exp)
    for i in range(len(exp)): 
	    if (exp[i] != "was" or exp[i] != "does"):
	      exp[i] = self.lemmatizer.lemmatize(exp[i])
    result = self.correct_spelling(exp)
    if(result != ' '): return result;  


  def normalize_and_tag(self, exp, deter=True, aux=True):
    """
	Parameters
	-----------
	exp : string
	   expression to normalize and tag

	deter : bool    
	   if true then determiners will be removed of expression

	aux : bool 
	   if true then auxiliary verbs in questions 
	   will be marked with "AUX" tag 

	Returns
	--------
	result : list
       list of lemmatized, tagged and correctly spelled words

	"""
    aux_verbs = ["am", "is", "are", "was", "were", "will", "did", "doe", "shall"]
    math_consts = ["π", "pi", "e"]
    exp = nltk.word_tokenize(exp)
    for i in range(len(exp)): 
      if (exp[i] != "was" or exp[i] != "does"):
	      exp[i] = self.lemmatizer.lemmatize(exp[i])
    exp = self.correct_spelling(exp)
    result = pos_tag(exp)
    for i in range(len(result)):
      if (result[i][0] in math_consts): 
        result[i] = list(result[i])
        result[i][1] = "NN"

    if (deter==True):
      _result = []
      for w in result:
        if(w[1] != "DT"): _result.append(w)
        result = _result;
    if (aux==True):
      if (result[0][0] in aux_verbs):
        result[0] = list(result[0])
        result[0][1] = "AUX"
    if(result != []):return result;


  def correct_spelling(self, exp):
    """
    Parameters
    -----------
    exp : string
      expression to correct

    Returns
    --------
    result : list
        list of corrected words

	  """
    misspelled = self.spell.unknown(exp)
    corrected = []
    corrected_words = [word for word in misspelled]
    for i in range(len(exp)):
      if (exp[i] not in misspelled):
	      corrected.append(exp[i])
      else:
        corrected.append(self.spell.correction(exp[i]))
        print("did you mean: ", self.spell.correction(exp[i]))
    return corrected;
