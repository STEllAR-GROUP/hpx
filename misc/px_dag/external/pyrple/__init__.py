#!/usr/bin/python
"""pyrple - A Python RDF API and Toolkit
Sean B. Palmer, 2003, <http://purl.org/net/sbp/>
"""

version = '2003-12-07'

# import rdf
import node
import triple
import graph
import parsers
from parsers import parse, ntriples, rdfxml
import namespaces
# import marshall

Node = node.Node
URI = node.URI
bNode = node.bNode
Literal = node.Literal
Var = node.Var
Triple = triple.Triple
Graph = graph.Graph
NTriples = ntriples.NTriples
Namespace = namespaces.Namespace

RDFSink = rdfxml.Sink
parseRDF = rdfxml.parseRDF
parseURI = rdfxml.parseURI

if __name__=="__main__": 
   print __doc__
