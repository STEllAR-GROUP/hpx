#!/usr/bin/python
"""triple.py - Triple class for Pyrple."""

from node import Node

class Triple(tuple): 
   def __new__(cls, *args): 
      if len(args) == 1: 
         if not len(args[0]) == 3: 
            raise "NonTripleError", "Not a triple: %s" % `args[0]`
         subj, pred, objt = tuple(args[0])
      elif len(args) == 3: 
         subj, pred, objt = tuple(args)
      else: raise "NonTripleError", "Not a triple: %s" % `args`
      for term in (subj, pred, objt): 
         assert (isinstance(term, Node) or term is None)
         #             or isinstance(term, Graph)
      return tuple.__new__(cls, (subj, pred, objt))

   def copy(self): 
      return Triple(self)

if __name__=="__main__": 
   print __doc__
