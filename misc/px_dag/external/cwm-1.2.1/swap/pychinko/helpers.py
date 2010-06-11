from terms import URI, Variable, Fact, Exivar
from types import IntType, DictType
from UserDict import UserDict
#from rdflib.TripleStore import TripleStore
import os
try:
  import cPickle as pickle
except ImportError:
  import pickle

output = open('pychinko.output.n3', 'w')

INCLUDES = "http://www.w3.org/2000/10/swap/log#includes"
IMPLIES = "http://www.w3.org/2000/10/swap/log#implies"
LOG_SEMANTICS = "http://www.w3.org/2000/10/swap/log#semantics"
LOG_CONJUNCTION = "http://www.w3.org/2000/10/swap/log#conjunction"
LOG_CONCLUSION = "http://www.w3.org/2000/10/swap/log#conclusion"

FIRST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first"
REST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"
NIL = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
LIST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#List"


def handleURL(filename):
  if filename.startswith('http://'):
    filename = 'tests/rdfig/' + filename.split('/')[-1]
  return filename

def bindingsp(object):
  """I check whether something is a binding returned by JOIN"""
  return isinstance(object, DictType)

def countTriples(filename):
  """I return the number of triples in `filename'."""
  print filename
  store = TripleStore()
  if filename.endswith('n3') or filename.endswith('ntriples'):
    # assume an ntriples file
    store.load(filename, format="nt")
  else:
    store.load(filename)
  return len(store)

def convertBNodeToFact( node):  
  if isinstance(node, Exivar):
    return node.name
  #elif isinstance(node, Variable):
    #return node.name
  else:
    return node


def loadRete(flo):
  """I unpersist a rete object from a file-like object."""
  u = pickle.Unpickler(flo)
  try:
    rete = u.load()
  except UnpicklingError:
    return False
  return rete

def storeRete(flo):
  """I persist a rete object to a file-like object (flo)."""
  p = pickle.Pickler(flo)
  p.dump(self)
  return flo

class Memory(dict):
  def __init__(self, dict=None):
    self.origInput = None


def aNodeListToPython(id, alphaNodes, py_list):
 
  if not alphaNodes:
    return py_list
  
  node_first = [node for node in alphaNodes if convertBNodeToFact(node.pattern.s)==id and node.pattern.p==FIRST]  

  """print "node_first:", node_first  
  if node_first:
          print "type:", type(node_first[0].pattern.o)
  print "fact_first:", fact_first
  if fact_first:
          print "type:", type(fact_first[0].o)"""
  #it's tricky here because the subj might be a variable
  if node_first:        
      py_list.append(node_first[0].pattern.o) # append variable
         
  node_rest = [node for node in alphaNodes if convertBNodeToFact(node.pattern.s)==id and node.pattern.p==REST]
  if node_rest:
    if node_rest[0].pattern.o == NIL:      
      return py_list
    else:          
      return aNodeListToPython(convertBNodeToFact(node_rest[0].pattern.o), alphaNodes, py_list)
  else:
      return py_list
  


def factListToPython(id, facts, py_list):
  if not facts:
    return py_list
  
  fact_first = [fact for fact in facts if convertBNodeToFact(fact.s)==id and fact.p==FIRST]

  if fact_first:
    py_list.append(fact_first[0].o) # append fact
     
  fact_rest = [fact for fact in facts if convertBNodeToFact(fact.s)==id and fact.p==REST]
  if fact_rest:
    if fact_rest[0].o == NIL:      
      return py_list
    else:          
      return factListToPython(convertBNodeToFact(fact_rest[0].o), facts, py_list)
  else:
      return py_list

#convert rdf list to python list and do NOT remove patterns associated with rdf_list
def listToPython(id, alphaNodes, facts, py_list):  

  if not alphaNodes:
    return py_list
  
  node_first = [node for node in alphaNodes if convertBNodeToFact(node.pattern.s)==id and node.pattern.p==FIRST]
  fact_first = [fact for fact in facts if convertBNodeToFact(fact.s)==id and fact.p==FIRST]

  """print "node_first:", node_first  
  if node_first:
          print "type:", type(node_first[0].pattern.o)
  print "fact_first:", fact_first
  if fact_first:
          print "type:", type(fact_first[0].o)"""
  #it's tricky here because the subj might be a variable
  if node_first:    
    if isinstance(node_first[0].pattern.o, Variable):
      py_list.append(node_first[0].pattern.o) # append variable
    else:
      py_list.append(fact_first[0].o) # append fact
  elif fact_first:
    py_list.append(fact_first[0].o) # append fact
     
  node_rest = [node for node in alphaNodes if convertBNodeToFact(node.pattern.s)==id and node.pattern.p==REST]
  if node_rest:
    if node_rest[0].pattern.o == NIL:      
      return py_list
    else:          
      return listToPython(convertBNodeToFact(node_rest[0].pattern.o), alphaNodes, facts, py_list)
  else:
      return py_list
  

def isListVarNode(node):
  return isinstance(node.pattern.s, Exivar) and node.pattern.p == FIRST and isinstance(node.pattern.o, Variable)

  
def cmpVar(x, y):
  if x.name < y.name:
    return -1
  elif x.name == y.name:
    return 0
  elif x.name > y.name:
    return 1

def sortVars(vars):
  """I am a destructive function."""
  return vars.sort(cmpVar)


def removedups(lst):
  nodups = list()
  for elt in lst:
    if not elt in nodups:
      nodups.append(elt)
  return nodups

def getOccurences(var, lst):
  """Return all positions of var in list."""
  loc = 0
  occurs = list()
  for elt in lst:
    if var == elt:
      occurs.append(loc)
    loc += 1
  return tuple(occurs)

def keysToList(dictionary):
  """I return a list of all keys in a nested dictionary, with optional
     values `prefix' inserted in the beginning."""  
  result = []
  justification = []
  if not dictionary:
    return result
  keysToListHelper(dictionary, [], result, justification)  
  return result

###NOTE--this function should be renamed ('memoryToList'?)
def keysToListHelper(dictionary, prefix, result, justification):
  """I take a dictionary representing a memory and convert it to a list based format, i.e.:
     [[v1, v2, v3, v4], [Fact(a, b, c), Fact(d, e, f)]
      ...]
     where the first list corresponds to a row in the memory, and the second to a
     set of facts justifying them."""
  if isinstance(dictionary, tuple) or not dictionary:
    # get the justification
    #for j in dictionary:
    #  justification.append(j)
    result.append([prefix, []])
    return 
  for k in dictionary:
    if len(dictionary[k]) > 1 and (not isinstance(dictionary[k], tuple)):
      for k2 in dictionary[k]:
        keysToListHelper(dictionary[k][k2], prefix + [k] + [k2], result, justification)
    else:
      keysToListHelper(dictionary[k], prefix + [k], result, justification)

if __name__ == "__main__":
  a = {'immanuel': {'alan': {},
                    'foo': {},
                    'grups': {}},
       'friedrich': {'drew': {}},
       'john': {'willard': {}},
       'willard': {'pat': {}},
       'richard': {'bertrand': {}}}

  print keysToList(a)

