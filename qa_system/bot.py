import re
import Analyzer
from deeppavlov import build_model, configs

class Bot:
  """
  Attributes:
  -----------

  base : wrapper for Wikipedia API
    returns wiki-pages

  model : bert model for Q&A

  analyzer : class for handling text data 

  """
  def __init__(self):
    self.base = wikipediaapi.Wikipedia('en')
    self.model = build_model(configs.squad.squad_bert, download=True)
    self.analyzer = Analyzer()
    

  def take_query(self):
    """
    Attributes
    ------------

    query : string
      user's input
    
    qtns : list
      questions extracted from query

    ents : list
      entities extracted from questions

    cntxt : list
      wiki-pages 

    """
    query = input("query: ")
    qtns = self.exract_questions(query)
    qtns = self.solve_anaphora(qtns)
    ents = self.extract_entities(qtns)
    cntxt = self.make_requests(ents)
    self.response(qtns, cntxt, ents)
  
  def response(self, qtns, contxts, ents):
    for i in range(len(contxts)):
        contxt = ' ';
        for j in range(len(contxts[i])):
          contxt += re.sub(r'\n|  |\{[^)]*\}|\\|\([^)]*\)|\S*\)|\S*\}', '',contxts[i][j].summary)
        answer = self.model([contxt], [qtns[i]])
        print("CONTEXT:", contxt)
        print("QUESTION:", qtns[i])
        if (answer[0][0] != ''):
          print("ANSWER:", answer[0][0])
        else:
          print("nothing about it")
        print("\n")
  

  def extract_entities(self, qtns):
    
    """
    Parameters
    -----------
    qtns : list of strings
      questions 

    Returns 
    --------
    result : list of string
      entities extracted from questions

    """

    result = []

    # proprocessing: normalize and tag
    tagged_qtns = []
    for qtn in qtns:
      qtn = self.analyzer.normalize_and_tag(qtn)
      tagged_qtns.append(qtn);

    #find all noun phrases in questions
    ####################################
    result = []
    for qt in tagged_qtns:
      ents = []
      i = 0;
      while (i < len(qt)):
        curr = ' ';
        if((qt[i][1] == "NN" or qt[i][1] == "JJ") and 
           (qt[i-1][1] != "WRB" and qt[i-1][1] != "WP" and qt[i-1][1] != "WDT") and i < len(qt)):
          curr += qt[i][0]+" "
          i += 1
          if (i < len(qt)):
            while(qt[i][1] == "NN" or qt[i][1] == "IN" or qt[i][1] == "JJ"):
              curr += qt[i][0]+" "
              print(curr)
              if (i == len(qt)-1): break;
              else: i+=1;
        if (curr != ' '): ents.append(curr)
        i += 1
      result.append(ents)
   
    
    # formate data
    ##################################### 
    formated_result = []
    for a in result:
      c = []
      for b in a:
        if(self.find_entity(b.strip(' ')) != "_" and 
           self.find_entity(b.strip(' ')) != None):
          c.append(self.find_entity(b.strip(' ')))
      formated_result.append(c)
    
    return formated_result


  def exract_questions(self, query):
    """
    Parameters
    -----------
    query : string
      query from user 

    Returns 
    --------
    result : list of string
      questions extracted from questions

    """
    
    # remove '!' and repetitive '?'
    query = re.sub(r'[!]','',query)
    query = re.sub(r'[?](?=\?)','',query)


    # scinario 1: if '?' is presented in query more then 1 times
    # then split string with it
    ########################################## 
    if (query.count('?') > 1):
      query = query.split("?")
      for i in range(len(query)):
        curr_qstn = ' ';
        query[i] = self.analyzer.normalize(query[i])
        for j in range(len(query[i])): 
          curr_qstn += query[i][j] + ' ';
        if (curr_qstn != ' '): result.append(curr_qstn);
      return result
    
    # scinario 2: '?' is not presented in query
    # then split string with wh-words and auxiliary verbs
    else:
      query = self.analyzer.normalize_and_tag(query)
      i = 0
      while (i<len(query)-1):
        if (query[i][1] == "WP" or query[i][1] == "WRB" or query[i][1] == "AUX"):
          curr_qstn= query[i][0]
          i += 1;
          while (query[i][1] != "WP" and query[i][1] != "WRB" and query[i][1] != "AUX"):
            curr_qstn += " " + query[i][0];
            if (i == len(query) - 1): break;
            else: i += 1;
          result.append(curr_qstn)
      return result
        

  def make_requests(self, ents):
    """
    Parameters
    -----------
    ents : list of strings
      entities extracted from user's query 

    Returns 
    --------
    result : list of custom wiki objects
      wiki pages that are related to extracted entities 

    """

    result = []

    for i in range(len(ents)):
      curr = []
      for j in range(len(ents[i])):
        # search only in realm of maths
        request = [ents[i][j], ents[i][j]+"_(mathematics)", ents[i][j]+"_(geometry)"]
        cntxt = [self.base.page(request[0]), self.base.page(request[1]), self.base.page(request[2])]
        if (cntxt[1].exists()):
          curr.append(cntxt[1])
        if (cntxt[2].exists()):
          curr.append(cntxt[2])
        else:
          curr.append(cntxt[0]) 
      result.append(curr)
    return result

  
 
  def find_entity(self, np):
    """
    Parameters
    -----------
    np : string
      noun phrase that may contain entity  

    Returns 
    --------
    ents : list of string
      entities extracted from the given noun phrase

    """

    # scinario 1: page with name 'np' exists on Wikipadia in math section
    if (self.base.page(np+"_(mathematics)").exists()):
      return np

    # scinario 2: page with name 'np' exists on Wikipedia and contains certain words
    elif (self.base.page(np).exists()):
      if ("mathematics" in self.base.page(np).text or
          "algebra" in self.base.page(np).text or
          "calculus" in self.base.page(np).text):
        return np

    # scinario 3: noun phrase contains preposition 'of' (e.g. result of cross product) 
    elif ('of' in np):
      new_np = np.split('of')[1]
      return self.find_entity(new_np)

    # scinario 4: noun phrase np consists of 3 words (e.g. cross product commutative)    
    elif (len(np.split(' ')) == 3):
      if (self.base.page(' '.join(np.split(" ")[1:])).exists()):
        return self.find_entity(' '.join(np.split(" ")[1:]))
      elif (self.base.page(' '.join(np.split(" ")[:2])).exists()):
        return self.find_entity(' '.join(np.split(" ")[:2]))

    # scinario 5: noun phrase np consists of 2 words (e.g. lines perpendicular)
    elif (len(np.split(' ')) == 2):
      if (self.base.page(np.split(" ")[0]).exists()):
        return self.find_entity(np.split(" ")[0])
      elif (self.base.page(np.split(" ")[1]).exists()):
        return self.find_entity(np.split(" ")[1])   
    
    else: return("_")
  
  def solve_anaphora(self, qnts):
    """
    takes questions, if it finds pronouns in some question
    then it tries to extract entity from previous question
    and replace the pronoun in the current one with it. 
    
    returns list of questions where all pronouns are replaced with
    certain entities
    """
    result = []
    for i in range(len(qnts)):
      curr = qnts[i].split(" ")
      for j in range(len(curr)):
        if (curr[j] == 'it' or curr[j] == 'they' or curr[j] == 'he'):
          ent = self.extract_entities([qnts[i-1]])
          curr[j] = 'a ' + ent[0][0]
      result.append(" ".join(curr))
    return result