#!/usr/bin/env python
"""ntriples.py - N-Triples parser for Pyrple."""

import re
import ntriplesg

class NTriples(object): 
   """An N-Triples parser with a customizable level of strictness.
      To override, set e.g.: NTriples.default = NTriples.spec
      or pass it along on __init__: NTriples(schema=NTriples.spec)
   """

   uri = ntriplesg.uri
   uriview = ntriplesg.uriview
   bnode = ntriplesg.bnode
   var = ntriplesg.var
   lang = ntriplesg.lang
   lit = ntriplesg.lit

   spec = ntriplesg.spec
   loose = ntriplesg.loose
   default = ntriplesg.default

   triple = ntriplesg.triple
   r_comment = ntriplesg.r_comment
   r_literal = ntriplesg.r_literal

   def __init__(self, sink=None, schema=None): 
      if sink is not None: self.sink = sink
      else: self.sink = []
      schema = ntriplesg.makeSchema(schema or self.default)
      self.r_triple = re.compile(self.triple % schema)

   def parseString(self, s, t=None, scrape=None): 
      """t is the function used to make a triple. scrape ignores errors."""
      t = t or tuple
      s = s.replace('\r\n', '\n').replace('\r', '\n')

      for line in s.splitlines(): 
         # assert not ntriplesg.r_disq.match(line), `line`
         m = self.r_triple.match(line)
         if m: self.sink.append(t(m.groups()))
         elif not line or not line.strip(' \t'): continue
         elif self.r_comment.match(line): continue
         elif not scrape: raise "ParseError", "line: %s" % line

      return self.sink
