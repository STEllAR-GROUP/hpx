#!/usr/bin/env python
"""parse.py - Parsing utilities for pyrple."""

import sys, os, re, urllib
import cPickle as pickle

# Should we cache by default or not?
# cache = '~/.pyrple/cache/'
cache = False

# Tools to get content from the Web

def get(uri, v=None): 
   # @@ send an accept: application/rdf+xml header
   u = urllib.urlopen(uri) # @@ urllib2
   info = u.info()

   contentType = info.get('Content-Type')
   lastModified = info.get('Last-Modified')
   contentLength = info.get('Content-Length')

   s = None
   global cache
   if cache: cache = os.path.expanduser(cache)

   if cache: 
      if not os.path.exists(cache): 
         os.makedirs(cache)

      # get a cached version if there is one
      fn = urllib.quote(uri, safe='')
      if os.path.exists(cache + fn) and contentType and lastModified: 
         cached = pickle.load(open(cache + fn, 'rb'))
         if cached[:3] == (contentType, lastModified, contentLength): 
            if v: print >> sys.stderr, "Using cached data: %s" % lastModified
            s = cached[3]
      elif os.path.exists(cache + fn): 
         if v: print >> sys.stderr, "Removing cache: new version uncacheable"
         os.remove(cache + fn) # @@

   # since there's no cached data, read from the Web
   if s is None: 
      if not contentLength: s = u.read()
      else: s = u.read(int(contentLength))
      u.close()

      if cache and contentType and lastModified: 
         if v: print >> sys.stderr, "Caching version: %s" % lastModified
         cached = (contentType, lastModified, contentLength, s)
         pickle.dump(cached, open(cache + fn, 'wb'))
   else: u.close()

   return (contentType, s)

# Tools to guage the serialization type

RDFXML = 'application/rdf+xml'
N3 = 'application/n3'
NTRIPLES = 'application/n-triples'
HTML = 'text/html'

r_xml = re.compile(r'^[\t\r\n ]*(<[?!]|<[^ >]+ )')

def getType(contentType, s): 
   """Guess the RDF serialization type of the input."""
   if contentType is None: 
      contentType = ''

   result = None
   if contentType.startswith('application/rdf+xml'): 
      return RDFXML
   elif contentType.endswith('notation3') or contentType.endswith('n3'): 
      return N3
   elif contentType in ('text/xml', 'application/xml'): 
      # @@ check root namespace? option to extract if transparent?
      return RDFXML
   elif (contentType.startswith('text/html') or 
         contentType.startswith('application/xhtml+xml')): 
      return HTML
   elif (contentType.endswith('ntriples') or 
         contentType.endswith('n-triples') or 
         contentType.endswith('nt')): 
      return NTRIPLES
   elif contentType in ('text/plain', 'application/octet-stream', ''): 
      if r_xml.match(s): 
         result = RDFXML
      else: 
         try: from ntriplesg import isNTriples
         except ImportError: result = N3
         else: 
            if not isNTriples(s): 
               result = N3
            else: result = NTRIPLES
   else: raise "NotImplemented", "Unknown MIME type: %s" % contentType

   return result

# Tools to extact RDF from HTML

r_rdfxml = re.compile(r'(?sm)(<(([^\s:]+:)?)RDF .+?</\2RDF>)')

def parseFromHTML(s, base=None): 
   if base is None: base = ''
   result = []

   data = [groups[0] for groups in r_rdfxml.findall(s)]
   if data: 
      for rdfxml in data: 
         result.append((base, RDFXML, rdfxml))
   return result

if __name__=="__main__": 
   print __doc__
