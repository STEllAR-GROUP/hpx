#! /usr/bin/python
"""
  The only way I can find of using rdflib's parser, and our store.
  It seems the RDFlib people agree --- just use the store's own types,
  With a usual dispatch to convert them.

  

  $ID   $
"""
try:
    from rdflib.syntax.parser import ParserDispatcher
except ImportError:
    from rdflib.syntax.parsers.RDFXMLParser import RDFXMLParser
    class ParserDispatcher(object):
        def __init__(self, sink):
            self.sink = sink
            self.realParser = RDFXMLParser()
        def __call__(self, source, format):
            self.realParser.parse(source, self.sink)
from rdflib.StringInputSource import StringInputSource
from rdflib.URIRef import URIRef
from rdflib.BNode import BNode
from rdflib.Literal import Literal
from RDFSink import FORMULA, ANONYMOUS, SYMBOL
import diag
from diag import progress


class rdflib_handoff:
    """There is a better way of doing this"""
    def __init__(self, store, openFormula, thisDoc, flags="", why=None):
        self.parser = ParserDispatcher(self)
        self.store = store
        self.format = 'xml'
        self.formula = openFormula
        self.asIfFrom = thisDoc
        self.ns_prefix_map = {}
        self.prefix_ns_map = {}
        self.anonymousNodes = {}
        self._reason = why      # Why the parser w
        self._reason2 = None    # Why these triples
        if diag.tracking: self._reason2 = BecauseOfData(sink.newSymbol(thisDoc), because=self._reason)

    def prefix_mapping(self, prefix, uri, override=False):
        self.prefix_ns_map[prefix] = uri
        self.ns_prefix_map[uri] = prefix
#        print 'why was I told about: ', prefix, uri
#        raise RuntimeError(prefix, prefix.__class__, uri, uri.__class__)
    bind = prefix_mapping
    
    def feed(self, buffer):
        self.parser(StringInputSource(buffer), self.format)

    def add(self, (subject, predicate, object)):
#        print subject, ", a ", type(subject)
#        print '---- has the property of ', predicate, ', of type ', type(predicate)
#        print '---- with the value of ', object, ', of type ', type(object), '.'
        self.store.makeStatement((self.formula,
               self.convertRDFlibTypes(predicate),
               self.convertRDFlibTypes(subject),
               self.convertRDFlibTypes(object)), self._reason2)
        return self

    def close(self):
        for prefix, uri in self.prefix_ns_map.items():
            if prefix == None: prefix = ""
            if ':' not in uri:
                uri = self.asIfFrom + uri
#            print '=+++++++++++++=', uri, "is a ", prefix
            self.store.bind(prefix,uri)
        return self.formula
    def convertRDFlibTypes(self, s):
        lang = None
        dt = None
        if isinstance(s, Literal):
            what = s
            if s.language != '':
                lang = s.language
            if s.datatype != '' and s.datatype != None:
                dt = self.store.newSymbol(s.datatype)
        elif isinstance(s,BNode):
            try:
                what = self.anonymousNodes[s]
            except KeyError:
                self.anonymousNodes[s] = self.store.newBlankNode(self.formula,uri=s)
                what = self.anonymousNodes[s]
        elif ':' in s:
            what = (SYMBOL, s)
        else:  #if s[0] == '#':
            what = (SYMBOL, self.asIfFrom + s)
#        else:
#            what = s
        return self.store.intern(what,dt=dt, lang=lang)
