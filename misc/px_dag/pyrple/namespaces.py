#!/usr/bin/python
"""namespaces.py - Namespace utilities for Pyrple."""

from node import URI, Var

class Namespace(unicode): 
   def __getattr__(self, name): return URI(self + name)
   def __getitem__(self, item): return URI(self + item)

RDF = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
PYRPLE = Namespace('http://infomesh.net/2003/pyrple#')
RDFS = Namespace('http://www.w3.org/2000/01/rdf-schema#')
OWL = Namespace('http://www.w3.org/2002/07/owl#')
FOAF = Namespace('http://xmlns.com/foaf/0.1/')
DC = Namespace('http://purl.org/dc/elements/1.1/')
CC = Namespace('http://web.resource.org/cc/')
LOG = Namespace('http://www.w3.org/2000/10/swap/log#')
STRING = Namespace('http://www.w3.org/2000/10/swap/string#')

VAR = type('_', (unicode,), {
   '__getattr__': lambda self, s: Var(s)
})()

# Make a dictionary of all the namespaces declared
all, gvars = {}, dict(globals())
for key in gvars: 
   if type(gvars[key]) is Namespace: 
      all[key.lower()] = gvars[key]
del gvars

if __name__=="__main__": 
   print __doc__
