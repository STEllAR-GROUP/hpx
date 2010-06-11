"""
This rdflib2rdf module provides code for glueing rdflib's rdf/xml
parser to cwm.

STATUS:

First pass, partially working, some questions.


QUESTIONS:

Does the context need to change per bag? I have assumed for the moment
not, as it breaks (or complicates) the interface between the parser
and the store.

Is there already or is there any interest in a ContextSink interface
to simplify the interface between the parser and the store? If
desired, I would be happy to help with any refactors that would be
needed. To better describe what I am thinking, I have implemented a
ContextSink enough to serve the purpose needed for this module.

Is urlparse.urljoin from the Python standard library buggy? RDFLib
uses urlparse.urljoin and I noticed that the differences between it
and uriparse.join are causing the test cases to fail (export
PYTHONPATH=..;python2.3 retest.py -n -f regression.n3). For example:
    file:/devel/WWW/2000/10/swap/test/animal.rdf#
    vs.
    file:///devel/WWW/2000/10/swap/test/animal.rdf#

Not really a question but... RDFLib's RDF/XML parser at the moment
does not make it easy to get to the namespace binding information. And
the way sax2rdf currently calls bind it looks like there will be
issues if a prefix gets redefined. Here is a question re:
this... should passing the namespace information to the store be a
[mandatory] part of the interface? or does it cause to much grief in
practice not too?


TO RUN: (cwm with rdflib's rdf/xml parser)

Double check that you have rdflib-1.2.x installed :)
  * Download from http://rdflib.net/stable/
  * extract
  * cd root of distribution
  * python2.2 setup.py install

Change the following line in cwm.py from:
    return sax2rdf.RDFXMLParser(...
to:
    return rdflib2rdf.RDFXMLParser(...

--eikeon, http://eikeon.com/

"""

from RDFSink import FORMULA
import diag
from diag import progress

class ContextSink(object):
    def __init__(self, sink, openFormula,
                 thisDoc=None,  flags="", why=None):
        self.sink = sink
        assert thisDoc != None, "Need document URI at the moment, sorry"
        self.thisDoc = thisDoc
        self.formulaURI = formulaURI
        self._context = openFormula
        self._reason = why      # Why the parser w
        self._reason2 = None    # Why these triples
        if diag.tracking: self._reason2 = BecauseOfData(sink.newSymbol(thisDoc), because=self._reason)
            
        
    def newSymbol(self, uri):
        return self.sink.newSymbol(uri)

    def newLiteral(self, s):
        return self.sink.newLiteral(s)

    def newXMLLiteral(self, s):
        return self.sink.newXMLLiteral(s)

    def newBlankNode(self):
        return self.sink.newBlankNode(self._context)

    def makeStatement(self, (predicate, subject, object)):
        self.sink.makeStatement((self._context, predicate, subject, object), why=self._reason2)


        

                    
import uripath

from rdflib.syntax.parser import ParserDispatcher
from rdflib.URIRef import URIRef
from rdflib.Literal import Literal
from rdflib.BNode import BNode
from rdflib.URLInputSource import URLInputSource


class RDFXMLParser:
    def __init__(self, sink,
                 formulaURI, thisDoc, flags="", why=None):
        self.__sink = ContextSink(sink,formulaURI, thisDoc, flags, why)
        self.__bnodes = {}
        
    def __convert(self, t):
        """Convert from rdflib to cwm style term."""

        # For now, I have decided not to try and add factory arguments
        # or some other such mechanism to rdflib's parser... but
        # rather convert the resulting rdflib term objects to the
        # desired types.

        # It may just be the cleanest interface moving forward as
        # well... as it removes the need to have to worry about the
        # details of the information needing to be passed to the
        # various constructors... for example, does the literal
        # constructor take a value and a language... and what happens
        # to the interface when datatype gets added. So in effect it
        # seems just to be a higher level interface.
        
        if isinstance(t, URIRef):
            return self.__sink.newSymbol(str(t))
        elif isinstance(t, Literal):
#            return self.__sink.newLiteral(str(t))
            return self.__sink.newLiteral(t)   # t is subclass of unicode
        elif isinstance(t, BNode):
            bnodes = self.__bnodes                    
            if t in bnodes:
                return bnodes[t]
            else:
                bnode = self.__sink.newBlankNode()
                bnodes[t] = bnode
                return bnode
        else:
            raise Exception("Unexpected type")

    def add(self, (s, p, o)):
        """Add triple to the sink (rdflib.syntax.parser.Parser calls
        this method)."""
        subject = self.__convert(s)
        predicate = self.__convert(p)
        object = self.__convert(o)                    
        self.__sink.makeStatement((predicate, subject, object))

    def feed(self, buffer):
        self.parser(StringInputSource(buffer))

    def load(self, uri, baseURI=""):
        if uri:
            uri = uripath.join(baseURI, uri) # Make abs from relative
        source = URLInputSource(uri)                                        
        self.parse(source)



#ends
