#!/usr/bin/python
"""node.py - Node class for Pyrple."""

from parsers.ntriplesg import r_literal, quote, unquote

def parseLiteral(s, lang=None, dtype=None): 
   """Returns the value of s, along with its lang and dtype."""
   b = r_literal.match(s).groups()
   if b[1] is not None: s = b[1]
   elif b[9] is not None: s = b[9]
   else: raise "StringError", "Got: %s" % s
   lang = lang or b[4] or b[12]
   dtype = dtype or b[7] and Node(b[7]) or None
   return s, lang, dtype

class Node(unicode): 
   """A Node in an RDF Graph."""

   def __new__(cls, n, lang=None, dtype=None, nt=None, ntval=None): 
      """Return the appropriate subClass of Node."""
      if nt and ntval: 
         assert cls in (URI, bNode, Var, Literal)
         n, ntval, value = nt, ntval, n
      elif cls in (URI, bNode, Var, Literal) or isinstance(n, list): 
         if isinstance(n, list): cls, n = Literal, n[0]

         if cls is Literal: 
            ntval, value = quote(n), n
         else: ntval = value = n

         terms = {URI: '<%s>', bNode: '_:%s', Var: '?%s', Literal: '"%s"'}
         n = terms[cls] % ntval

         if cls is Literal: 
            if lang: n += '@' + lang
            elif isinstance(dtype, Node): # @@ if?
               n += '^^' + dtype
      elif cls is Node: 
         assert n and n[0] in '<_?"', "Invalid N-Triples term: %s" % n
         cls, val = {'<': (URI, slice(1, -1)), 
                     '_': (bNode, slice(2, None)), 
                     '?': (Var, slice(1, None)), 
                     '"': (Literal, slice(1, -1))}[n[0]]

         if (cls is Literal) and (not n.endswith('"')): 
            ntval, lang, dtype = parseLiteral(n, lang, dtype)
         else: ntval = n[val]

         if cls is Literal: 
            value = unquote(ntval)
         else: value = ntval
      else: raise "UnknownClassError", "Unknown class: %s" % cls

      if cls is not Literal: 
         assert not (lang or dtype)
      elif (lang and dtype): 
         raise "LangAndDtypeError", "lang and dtype are exclusive"

      r = unicode.__new__(cls, n)
      r.ntval, r.value = ntval, value
      if lang: r.lang = lang
      elif dtype: r.dtype = dtype
      return r

   __repr__ = lambda self: self
   __str__ = lambda self: self.ntval

class URI(Node): pass
class bNode(Node): pass
class Var(Node): pass
class Literal(Node): pass

if __name__=="__main__": 
   print __doc__
