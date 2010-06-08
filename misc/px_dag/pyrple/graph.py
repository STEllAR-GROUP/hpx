#!/usr/bin/env python
"""graph.py - Graph class for Pyrple"""

import re, cgi, copy, urllib

from node import Node, URI, bNode, Literal, Var
from triple import Triple
from namespaces import PYRPLE
from util import label
from parsers.ntriples import NTriples

qvars = (Var,)
useBuiltins = True

class Graph(list): 
   def __init__(self, **kargs): 
      self.triples = {}
      self.patterns = {}
      self.hash = None

      args = ('node', 'uri', 'triples', 'ntriples', 'rdfxml', 'n3')
      feeds = {'node': self.feedNode, 
               'uri': self.feedURI, 
               'triples': self.feedTriples, 
               'ntriples': self.feedNTriples, 
               'rdfxml': self.feedRDFXML, 
               'n3': self.feedN3}

      for karg in kargs.keys(): 
         if karg in args: 
            if karg == 'triples' and kargs[karg]: 
               if isinstance(kargs[karg][0], Triple): 
                  kargs[karg] = [kargs[karg]]
            elif not isinstance(kargs[karg], list): 
               kargs[karg] = [kargs[karg]]
            elif kargs[karg] is None: kargs[karg] = []
         else: del kargs[karg]

      for arg in args: 
         if kargs.has_key(arg): 
            feeds[arg](*tuple(kargs[arg]))

   def __add__(self, G): 
      """Add two graphs; merge them."""
      F = self.copy()
      F += G
      return F

   def __iadd__(self, G): 
      """Merge another graph into this one."""
      self.merge(G)
      return self

   def __eq__(self, G): 
      """Graph isomorphism testing."""
      if not isinstance(G, Graph): return False
      elif len(self) != len(G): return False
      elif list.__eq__(self, G): return True # @@
      return hash(self) == hash(G)

   def __ne__(self, G): 
      """Negative graph isomorphism testing."""
      return not self.__eq__(G)

   def __hash__(self): 
      if self.hash: 
         if list.__eq__(self, self.hash[1]): 
            return self.hash[0]

      def hashVariable(G, term, done=None): 
         triple_hashes = []
         for triple in G.triples.keys(): 
            if term in triple: 
               term_hashes = []

               for p in xrange(3): 
                  if not (type(triple[p]) in (bNode, Var)): 
                     term_hash = hash(triple[p])
                  elif done or triple[p] == term: term_hash = p
                  else: term_hash = hashVariable(G, triple[p], done=1)
                  term_hashes.append(term_hash)

               triple_hashes.append(hash(tuple(term_hashes)))

         triple_hashes.sort()
         return hash(tuple(triple_hashes))

      triple_hashes = []
      for triple in self.triples.keys(): 
         xyz = [(type(term) in (bNode, Var) and hashVariable(self, term)) 
                 or hash(term) for term in triple]
         triple_hashes.append(hash(tuple(xyz)))
      triple_hashes.sort()

      result = hash(tuple(triple_hashes))
      self.hash = (result, self) # @@ copy of self? hmm
      return result

   def __contains__(self, t): 
      if self.__tquery(t): return True
      else: return False

   # Feed methods

   def feedNode(self, *args): 
      for node in args: 
         if type(node) is URI: 
            self.feedURI(node.value)
         elif type(node) is Literal: 
            if node.dtype == PYRPLE.NTriples: 
               self.feedNTriples(node.value)
            elif node.dtype == rdf.XMLLiteral: 
               self.feedRDFXML(node.value)

   def feedURI(self, *args): 
      from parsers import parse
      for uri in args: 
         contentType, s = parse.get(uri)
         t = parse.getType(contentType, s)
         if t == parse.HTML: 
            n = parse.parseFromHTML(s, base=uri)
         else: n = [(uri, t, s)]

         for (uri, t, s) in n: 
            if t == parse.RDFXML: 
               self.feedRDFXML(s, base=uri)
            elif t == parse.NTRIPLES: 
               self.feedNTriples(s, base=uri)
            elif t == parse.N3: 
               self.feedN3(s, base=uri)

   def feedTriples(self, *args, **kargs): 
      base = kargs.get('base')
      for triples in args: 
         if len(self) == 0: 
            for triple in triples: 
               self.append(triple)
         else: self.merge(Graph(triples=triples))

   def feedNTriples(self, *args, **kargs): 
      base = kargs.get('base')
      def parseNTriples(s): 
         def triple(spo): return Triple(*[Node(t) for t in spo])
         return NTriples().parseString(s, t=triple)

      for ntriples in args: 
         if len(self) == 0: 
            for triple in parseNTriples(ntriples): 
               self.append(triple)
         else: self.merge(Graph(ntriples=ntriples))

   r_xml = re.compile(r'^[\t\r\n ]*(<[?!]|<[^ >]+ )')

   class __RDFSink(object): 
      def __init__(self): 
         self.graph = Graph()
      def triple(self, s, p, o): 
         self.graph.append(Triple(Node(s), Node(p), Node(o)))

   def __parseRDF(self, s, **kargs): 
      base = kargs.get('base')
      try: from parsers import rdfxml
      except ImportError: rdfxml = None
      G = rdfxml.parseRDF(s, base=None, sink=self.__RDFSink()).graph
      return G

   def feedRDFXML(self, *args, **kargs): 
      base = kargs.get('base')
      for rdfxml in args: 
         G = self.__parseRDF(rdfxml, base=base)
         if len(self) == 0: 
            self.extend(G)
         else: self.merge(G)

   def feedN3(self, *args, **kargs): 
      base = kargs.get('base')
      try: from parsers import n3
      except ImportError: n3 = None
      for s in args: 
         triples = n3.parseN3(s, base=base).triples
         if len(self) == 0: 
            for triple in triples: 
               self.append(triple)
         else: self.merge(Graph(triples=triples))

   # Graph augmentation methods

   maps = ((0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0))

   def append(self, triple, force=False): 
      """Append a triple *without* merging."""
      assert isinstance(triple, Triple)

      if force or (not self.triples.has_key(triple)): 
         self.triples[triple] = 0
         for m in self.maps: 
            t = [((m[p] == 1 and triple[p]) or None) for p in xrange(3)]
            self.patterns.setdefault(Triple(t), {})[triple] = 0
         list.append(self, triple)

   def extend(self, triples): 
      """Extend some triples *without* merging."""
      for triple in triples: 
         self.append(triple)

   def add(self, triple): 
      """Add a triple and merge. Use += to add more."""
      self.merge(Graph(triples=[triple]))

   def remove(self, t): 
      """Removes a triple as-is, no querying."""
      # self.infer(triples, [], mode='replace')
      if type(t) is Triple: 
         triples = [t]
      else: triples = t

      for triple in triples: 
         if self.triples.has_key(triple): 
            list.remove(self, triple)
            del self.triples[triple]
            for m in self.maps: 
               t = [((m[p] == 1 and triple[p]) or None) for p in xrange(3)]
               del self.patterns[Triple(t)][triple]
         else: raise "TripleNotInGraphError", "Got: %s" % triple

   def merge(self, G): 
      """Merge another graph into this one, renaming variables as you go.
         This is slow because of variables. If you are sure that your 
         graph is grounded, use the extend method instead.
      """
      import time
      t = str(time.time()).replace('.', '')
      label = 'id%s' % abs(id(G) ^ int(t)) # @@ add hash to label?
      for triple in G: 
         # if i % 10000 == 0: 
         #    print >> sys.stderr, "Done %s triples (%s)" % (i, len(self))
         foundvar, new_triple = 0, []

         for p in xrange(3): 
            if type(triple[p]) is bNode: 
               foundvar = 1
               new_triple.append(bNode(label + triple[p].value))
            elif type(triple[p]) is Var: 
               foundvar = 1
               new_triple.append(Var(label + triple[p].value))
            else: new_triple.append(triple[p])

         if not foundvar: self.append(triple)
         else: self.append(Triple(new_triple))

   # Graph utility functions (serialization, copying, and isomorphism)

   def toRDFXML(self): 
      r_xmlid = re.compile(r'^[A-Za-z_][A-Za-z0-9._-]*$')
      RDF_NS = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
      rdf = '<rdf:RDF xmlns:rdf="%s">\n' % RDF_NS
      for (s, p, o) in self: 
         attrs = {URI: 'rdf:about', bNode: 'rdf:nodeID'}
         try: rdf += '<rdf:Description %s="%s">\n' % (attrs[type(s)], s.value)
         except KeyError: raise Exception, "Can't output %s" % s

         i = (p.value.rfind('#') + 1)
         if i > 0: xmlns, name = p.value[:i], p.value[i:]
         else: 
            i = (p.value.rfind('/') + 1) or -1
            xmlns, name = p.value[:i], p.value[i:]
         if (type(p) is not URI) or (not r_xmlid.match(name)): 
            raise Exception, "Can't output %s" % p
         rdf += '   <%s xmlns="%s"' % (p.value[i:], p.value[:i])

         attrs[URI] = 'rdf:resource'
         try: rdf += ' %s="%s"/>\n' % (attrs[type(o)], o.value)
         except KeyError: 
            rdf += '>%s</%s>\n' % (cgi.escape(o.value), p.value[i:])
         rdf += '</rdf:Description>\n'
      return rdf + '</rdf:RDF>\n'

   def serialize(self, as=None): 
      if as in (None, 'ntriples'): 
         return '\n'.join(['%r %r %r .' % triple for triple in self]) + '\n'
      elif as == 'rdfxml': 
         return self.toRDFXML()
      elif as == 'literal': 
         return Literal(self.serialize(), dtype=PYRPLE.NTriples)
      else: raise "UnknownSerializationTypeError", "Got: %s" % as

   def copy(self): 
      # @@ copy.copy and copy.deepcopy don't work...
      G = Graph()
      for triple in self.triples.keys(): 
         G.append(triple)
      return G

   # Graph query methods

   def __tquery(self, t): 
      key = tuple([(type(t[p]) not in qvars and t[p]) or None 
             for p in xrange(3)])
      if key == (None, None, None): 
         return self.triples.keys()
      elif None not in key: 
         if self.triples.has_key(key): 
            return [key]
      elif self.patterns.has_key(key): 
         return self.patterns[key].keys()
      return []

   def get(self, subj, pred=None, objt=None): 
      if subj is None: subj = Var(label())
      if pred is None: pred = Var(label())
      if objt is None: objt = Var(label())

      t = Triple(subj, pred, objt)
      return Graph(triples=self.__tquery(t))

   def __fquery(self, G, bindings, found=None): 
      if len(G) > 0: 
         first, rest = G[0], G[1:]
         for result in self.__tquery(first): 
            b = copy.copy(bindings)
            if type(found) is Graph: f = found.copy()
            elif found is not None: f = copy.copy(found)
            else: f = None
            fail = 0
            for j in xrange(3): 
               term = first[j]
               if type(term) in qvars: 
                  if b[term] is None: b[term] = result[j]
                  elif b[term] != result[j]: 
                     fail = 1
                     break
            if not fail: 
               if f is not None: f.append(result)
               self.__fquery(rest, b, f)
      elif None not in bindings.values(): 
         for key in bindings.keys(): 
            if type(key) is bNode: del bindings[key]
         if found is None: self.result.append(bindings)
         else: self.result.append((bindings, found))

   def query(self, G, sink=None): 
      """Query the graph, returning a list of bindings and maybe results."""
      if sink is 1: sink = Graph()
      bindings, self.result = {}, []
      for triple in G: 
         for term in triple: 
            if type(term) in qvars: 
               bindings[term] = None
      self.__fquery(G, bindings, found=sink)
      return self.result

   def filter(self, G): 
      graphs = [solution[1] for solution in self.query(G, sink=1)]
      if len(graphs) > 1: 
         R = graphs[0]
         for graph in graphs[1:]: 
            R += graph
         return R
      elif len(graphs) == 1: return graphs[0]
      else: return Graph()

   def infer(self, X, Y, mode=None): 
      # @@ the mode changes whether or not this is an in-place method
      mode = mode or 'filter'
      Y_list = []
      if useBuiltins: 
         import builtins
         X, bi = builtins.getBuiltins(X)
      for (bindings, triples) in self.query(X, sink=[]): 
         if useBuiltins: 
            Y = builtins.doBuiltins(Y, bi, bindings)
         for triple in Y: 
            t = Triple([((type(term) is Var) and (bindings.get(term) or term)) 
                   or term for term in triple])
            Y_list.append(t)
         if mode == 'replace': 
            self.remove(triples)
      result = Graph(triples=Y_list)
      if mode == 'filter': 
         return result
      elif mode in ('apply', 'replace'): 
         self.extend(result) # @@ return self + result? @@ +=?
      else: raise "UnknownModeError", "Got: %s" % mode

   def getRules(self): 
      return [(Graph(ntriples=t[0].value), Graph(ntriples=t[2].value)) 
               for t in self.get(None, PYRPLE.implies, None)] # @@ log:implies?

   def think(self, mode=None): # @@ mode
      oldlen, newlen = -1, 0
      while (oldlen < newlen): 
         oldlen = len(self)
         for rule in self.getRules(): 
            self.infer(rule[0], rule[1], mode='apply')
         newlen = len(self)
         # import sys
         # print >> sys.stderr, "Found %s new statements" % (newlen - oldlen)

if __name__=="__main__": 
   print __doc__
