###
### Facilities for generating proofs
###

from terms import Pattern, URI
from helpers import removedups, sortVars, keysToList

class ProofTrace:
  """I represent a trace of the patterns that matched (i.e. alpha nodes) that justify
     a certain inference."""
  def __init__(self):
    self.trace = list()

  def __getitem__(self, index):
    return self.trace[index]

  def __len__(self):
    return len(self.trace)

  def addPremise(self, premise):
    self.trace.append(premise)
    
  def render(self):
    """I render the proof trace as a list"""
    counter = 1
    for premise in self.trace:
      print "Premise " + str(counter) + ": " + premise
      counter += 1
    
