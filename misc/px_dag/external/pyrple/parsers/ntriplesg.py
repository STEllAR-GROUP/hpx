#!/usr/bin/env python
"""ntriplesg.py - N-Triples grammar terms for Pyrple."""

# Condition: this module may only import from stdlib

import re

uri = r'<[^>]*>'
uriview = r'<[^:]+:[^>]+>'
bnode = r'_:[A-Za-z][A-Za-z0-9]*'
var = r'\?[A-Za-z][A-Za-z0-9]*'
lang = r'"(?:[^"\\]*(?:\\.[^"\\]*)*)"(?:@(?:[a-z0-9]+(?:-[a-z0-9]+)?))?'
lit = r'(?:' + lang + r'(?:\^\^(?:' + uri + ')))|' + '(?:' + lang + ')'

spec = ((uriview, bnode), (uriview,), (lit, uriview, bnode))
loose = ((lit, uri, bnode, var), (uri, bnode, var), (lit, uri, bnode, var))
default = loose

triple = r'^[ \t]*(%s)[ \t]+(%s)[ \t]+(%s)[ \t]*\.[ \t]*$'

def makeSchema(s): 
   return tuple(['%s' % '|'.join(['(?:%s)' % p for p in term]) 
                 for term in s])

r_triple = re.compile(triple % makeSchema(default))
r_comment = re.compile(r'^[ \t]*#')
r_literal = re.compile(lit.replace('?:', ''))

# N-Triples testing

def isNTriples(s): 
   s = s.replace('\r\n', '\n')
   s = s.replace('\r', '\n')
   for line in s.split('\n'): 
      if not (r_triple.match(line) or 
              r_comment.match(line) or 
              (not line.strip(' \t'))): 
         return False
   return True

# N-Triples string quoting
# Cf. http://www.w3.org/TR/rdf-testcases/#ntrip_strings

r_disq = re.compile(ur'(?<!\\)"|[\x00-\x1F\x7F-\uFFFF]')
r_uniupper = re.compile(r'(?<=\\u)([0-9A-F]{4})|(?<=\\U)([0-9A-F]{8})')
r_unilower = re.compile(r'(?<=\\u)([0-9a-f]{4})|(?<=\\U)([0-9a-f]{8})')
r_hibyte = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\xFF]')
r_escapes = re.compile(r'\\(?!\\|"|n|r|t|u[0-9A-F]{4}|U[0-9A-F]{8})')

def quote(s): 
   if type(s) is not unicode: s = unicode(s, 'latin-1') # @@ utf-8?
   s = s.replace('\\', r'\\') # unless u'\\'.encode('unicode-escape') = '\\\\'
   s = s.replace('"', r'\"')
   s = s.replace(r'\\"', r'\"')
   s = r_hibyte.sub(lambda m: '\\u00%02X' % ord(m.group(0)), s)
   s = s.encode('unicode-escape')
   s = r_unilower.sub(lambda m: (m.group(1) or m.group(2)).upper(), s)
   return str(s)

def unquote(s): 
   if r_disq.match(s) or r_unilower.match(s) or r_escapes.match(s): 
      raise "IllegalCharacterError", "Got: %r" % s
   for (u, U) in r_uniupper.findall(s): 
      if int(u or U, 16) > 0x10FFFF: 
         raise "IllegalCharacterError", "Found: %s" % U
   s = str(s).decode('unicode-escape')
   return unicode(s)

if __name__=="__main__": 
   print __doc__
