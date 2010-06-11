#!/usr/bin/env python
"""
N3P - An N3 Parser using n3.n3
Author: Sean B. Palmer, inamidst.com
Licence: GPL 2; share and enjoy!
Documentation: http://inamidst.com/n3p/
Derived from: 
   http://www.w3.org/2000/10/swap/grammar/predictiveParser.py
   - predictiveParser.py, Tim Berners-Lee, 2004
Issues: 
   http://lists.w3.org/Archives/Public/public-cwm-bugs/2005Jan/0006
   http://lists.w3.org/Archives/Public/public-cwm-talk/2005JanMar/0015
"""

import sys, os, re, urllib
import cPickle as pickle

try: set()
except NameError: 
   from sets import Set as set

try: 
   import n3meta
   branches = n3meta.branches
   regexps = n3meta.regexps
except ImportError: 
   for path in sys.path: 
      fn = os.path.join(path, 'n3meta.pkl')
      if os.path.isfile(fn): 
         f = open(fn, 'rb')
         n3meta = pickle.load(f)
         f.close()

         branches = n3meta['branches']
         regexps = n3meta['regexps']
         break

start = 'http://www.w3.org/2000/10/swap/grammar/n3#document'

r_whitespace = re.compile(r'[ \t\r\n]*(?:(?:#[^\n]*)?\r?(?:$|\n))?')
singleCharacterSelectors = "\t\r\n !\"#$%&'()*.,+/;<=>?[\\]^`{|}~"
r_qname = re.compile(r'([A-Za-z0-9_:]*)')
r_name = re.compile(r'([A-Za-z0-9_]*)')
notQNameChars = singleCharacterSelectors + "@"
notNameChars = notQNameChars + ":"

def abbr(prodURI): 
   return prodURI.split('#').pop()

class N3Parser(object): 
   def __init__(self, uri, branches, regexps):
      if uri == 'nowhere':
          pass
      else:
          if (uri != 'file:///dev/stdin'): 
             u = urllib.urlopen(uri)
             self.data = u.read()
             u.close()
          else: self.data = sys.stdin.read()
      self.pos = 0
      self.branches = branches
      self.regexps = regexps
      self.keywordMode = False
      self.keywords = set(("a", "is", "of", "this", "has"))
      self.productions = []
      self.memo = {}

   def parse(self, prod):
      todo_stack = [[prod, None]]
      while todo_stack:
          #print todo_stack
          #prod = todo_stack.pop()
          if todo_stack[-1][1] is None:
              todo_stack[-1][1] = []
              tok = self.token()
              # Got an opened production
              self.onStart(abbr(todo_stack[-1][0]))
              if not tok: 
                 return tok # EOF

              prodBranch = self.branches[todo_stack[-1][0]]
              sequence = prodBranch.get(tok, None)
              if sequence is None: 
                 print >> sys.stderr, 'prodBranch', prodBranch
                 raise Exception("Found %s when expecting a %s . todo_stack=%s" % (tok, todo_stack[-1][0], `todo_stack`))
              for term in sequence:
                 todo_stack[-1][1].append(term)
          while todo_stack[-1][1]:
             term = todo_stack[-1][1].pop(0)
             if isinstance(term, unicode): 
                j = self.pos + len(term)
                word = self.data[self.pos:j]
                if word == term: 
                   self.onToken(term, word)
                   self.pos = j
                elif '@' + word[:-1] == term: 
                   self.onToken(term, word[:-1])
                   self.pos = j - 1
                else: raise Exception("Found %s; %s expected" % \
                             (self.data[self.pos:self.pos+10], term))
             elif not self.regexps.has_key(term): 
                todo_stack.append([term, None])
                continue
             else: 
                regexp = self.regexps[term]
                m = regexp.match(self.data, self.pos)
                if not m: 
                   raise Exception("Token: %r should match %s" % \
                          (self.data[self.pos:self.pos+10], regexp.pattern))
                end = m.end()
                self.onToken(abbr(term), self.data[self.pos:end])
                self.pos = end
             self.token()
          while todo_stack[-1][1] == []:
              todo_stack.pop()
              self.onFinish()

   def token(self): 
      """Memoizer for getToken."""
      if self.memo.has_key(self.pos): 
         return self.memo[self.pos]
      result = self.getToken()
      pos = self.pos
      self.memo[pos] = result
      return result

   def getToken(self): 
      self.whitespace()
      if self.pos == len(self.data): 
         return '' # EOF!

      ch2 = self.data[self.pos:self.pos+2]
      for double in ('=>', '<=', '^^'): 
         if ch2 == double: return double

      ch = self.data[self.pos]
      if ch == '.' and self.keywordMode: 
         self.keywordMode = False

      if ch in singleCharacterSelectors + '"': 
         return ch
      elif ch in '+-0123456789': 
         return '0'

      if ch == '@': 
         if self.pos and (self.data[self.pos-1] == '"'): 
            return '@'
         name = r_name.match(self.data, self.pos + 1).group(1)
         if name == 'keywords': 
            self.keywords = set()
            self.keywordMode = True
         return '@' + name

      word = r_qname.match(self.data, self.pos).group(1)
      if self.keywordMode: 
         self.keywords.add(word)
      elif word in self.keywords: 
         if word == 'keywords': 
            self.keywords = set()
            self.keywordMode = True
         return '@' + word # implicit keyword
      return 'a'

   def whitespace(self): 
      while True: 
         end = r_whitespace.match(self.data, self.pos).end()
         if end <= self.pos: break
         self.pos = end

   def onStart(self, prod): 
      print (' ' * len(self.productions)) + prod
      self.productions.append(prod)

   def onFinish(self): 
      prod = self.productions.pop()
      print (' ' * len(self.productions)) + '/' + prod

   def onToken(self, prod, tok): 
      print (' ' * len(self.productions)) + prod, tok

def main(argv=None): 
   if argv is None: 
      argv = sys.argv
   if len(argv) == 2: 
      p = N3Parser(argv[1], branches, regexps)
      p.parse(start)

if __name__=="__main__": 
   main()
