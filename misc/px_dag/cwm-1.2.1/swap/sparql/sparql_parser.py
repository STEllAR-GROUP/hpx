#!/usr/bin/env python
"""
sparql_parser. A parser of SPARQL based on 
    N3P - An N3 Parser using n3.n3
    Author: Sean B. Palmer, inamidst.com
    Licence:    http://www.w3.org/Consortium/Lega l/copyright-software
    Documentation: http://inamidst.com/n3p/
    Derived from: 
       http://www.w3.org/2000/10/swap/grammar/predictiveParser.py
       - predictiveParser.py, Tim Berners-Lee, 2004

Sparql_parser is by Yosi Scharf, and quite heavily modified
from n3p
"""
try:
    Set = set
except NameError:
    from sets import Set

import sys, os, re, urllib
import sparql_tokens

tokens = Set()
import cPickle as pickle

try: 
   import sparql_table
   branches = sparql_table.branches
except ImportError: 
   for path in sys.path: 
      fn = os.path.join(path, 'sparql.pkl')
      if os.path.isfile(fn): 
         f = open(fn, 'rb')
         n3meta = pickle.load(f)
         f.close()

         branches = n3meta['branches']
         break

start = 'http://www.w3.org/2000/10/swap/grammar/sparql#Query'

def abbr(prodURI): 
   return prodURI.split('#').pop()

class N3Parser(object): 
   def __init__(self, buffer, branches, sink, ve=0):
      lexer = sparql_tokens.Lexer()
      lexer.input(buffer)
      tokens.update(sparql_tokens.tokens)
      self.data = lexer.token
      self.newToken()
      self.branches = branches
      self.productions = []
      self.memo = {}
      self.sink = sink
      self.verbose = ve

   def parse(self, prod):
      todo_stack = [[prod, None]]
      while todo_stack:
          #print todo_stack
          #prod = todo_stack.pop()
          if todo_stack[-1][1] is None:
              todo_stack[-1][1] = []
              tok = self.token
              # Got an opened production
              self.onStart(abbr(todo_stack[-1][0]))
              if not tok: 
                 return tok # EOF

              try:
                  prodBranch = self.branches[todo_stack[-1][0]]
              except:
                  print todo_stack
                  print self.branches.keys()
                  raise
              #print prodBranch
              sequence = prodBranch.get(tok[0], None)
              if sequence is None: 
                 print >> sys.stderr, 'prodBranch', prodBranch
                 raise Exception("Found %s when expecting a %s . todo_stack=%s" % (tok, todo_stack[-1][0], `todo_stack`))
              for term in sequence:
                 todo_stack[-1][1].append(term)
          while todo_stack[-1][1]:
             term = todo_stack[-1][1].pop(0)
             if term in tokens: 
                name, word, line = self.token
                if name == term: 
                   self.onToken(term, word)
                   self.newToken()
                else: raise Exception("Found %s; %s expected" % \
                             (`self.token`, term))
             else:
                todo_stack.append([term, None])
                
          while todo_stack and todo_stack[-1][1] == []:
              todo_stack.pop()
              if not todo_stack:
                  return self.onFinish()
              self.onFinish()
      

   def newToken(self): 
      self.token = self.data()

   def onStart(self, prod):
      if self.verbose:
          print (' ' * len(self.productions)) + `prod`
      self.productions.append([prod])

   def onFinish(self): 
      prod = self.sink.prod(self.productions.pop())
      if self.productions:
          self.productions[-1].append(prod)
      if self.verbose:
          print (' ' * len(self.productions)) + '/' + `prod`
      return prod

   def onToken(self, prod, tok):
      self.productions[-1].append((prod, tok))
      if self.verbose:
          print (' ' * len(self.productions)) + `(prod, tok)`

class nullProductionHandler(object):
    def prod(self, production):
        return production[0]

def main(argv=None):
   if argv is None: 
      argv = sys.argv
   from swap import sparql2cwm, myStore, notation3
   _outSink = notation3.ToN3(sys.stdout.write,
                                      quiet=1, flags='')
   sink = sparql2cwm.FromSparql(myStore._checkStore(), ve=1)
   if len(argv) == 3:
       sink = nullProductionHandler()
       p = N3Parser(file(argv[1], 'r'), branches, sink, ve=1)
       p.parse(start)
   if len(argv) == 2: 
      p = N3Parser(file(argv[1], 'r'), branches, sink, ve=1)
      print "About to parse"
      f = p.parse(start).close()
      myStore._checkStore().dumpNested(f, _outSink)

if __name__=="__main__": 
   main()
