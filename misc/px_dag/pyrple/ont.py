#!/usr/bin/env python
"""Ont module for pyrple."""

import triple, graph
from namespaces, VAR, RDF, RDFS, OWL, THIS

# RDF Stuff

def isTyped(G, term): 
   Q = graph.Graph(triples=[triple.Triple(term, RDF.type, VAR.x)])
   return len(G.query(Q)) > 0

def isProperty(G, term): 
   return pyrple.triple.Triple(term, RDF.type, RDF.Property) in G

def getList(G, term): 
   """Get a list from G returned as a Python list of Nodes."""
   result = []
   if term == RDF.nil: 
      return result
   while 1: 
      bindings = G.query(Graph(triples=[
         Triple(term, RDF.first, VAR.first), 
         Triple(term, RDF.rest, VAR.rest)
      ]))
      if len(bindings) != 1: return False
      first = bindings[0][VAR.first]
      rest = bindings[0][VAR.rest]
      result.append(first)
      if rest == RDF.nil: 
         break
      term = rest
   return result

# RDFS Stuff

def isRDFSClass(G, term): 
   return triple.Triple(term, RDF.type, RDFS.Class) in G

def getDomains(G, term): 
   Q = graph.Graph(triples=[triple.Triple(term, RDFS.domain, VAR.x)])
   return [result[VAR.x] for result in G.query(Q)]

def getRanges(G, term): 
   Q = graph.Graph(triples=[triple.Triple(term, RDFS.domain, VAR.x)])
   return [result[VAR.x] for result in G.query(Q)]

def isDatatype(G, term): 
   # @@ RDFS?
   return triple.Triple(term, RDF.type, RDFS.Datatype) in G

def getSuperProperties(G, term): 
   Q = graph.Graph(triples=[triple.Triple(term, RDFS.subPropertyOf, VAR.x)])
   return [binding[VAR.x] for binding in G.query(Q)]

def getSuperClasses(G, term): 
   Q = graph.Graph(triples=[triple.Triple(term, RDFS.subClassOf, VAR.x)])
   return [binding[VAR.x] for binding in G.query(Q)]

def getSubProperties(G, term): 
   Q = graph.Graph(triples=[triple.Triple(VAR.x, RDFS.subPropertyOf, term)])
   return [result[VAR.x] for result in G.query(Q)]

def getSubClasses(G, term): 
   Q = graph.Graph(triples=[triple.Triple(VAR.x, RDFS.subClassOf, term)])
   return [result[VAR.x] for result in G.query(Q)]

# OWL Stuff

def disjoint(G, p, q): 
   if (triple.Triple(p, OWL.disjointWith, q) in G or
       triple.Triple(q, OWL.disjointWith, p) in G): 
      return True
   return False

# def cardinality(G, cls, prop): 
#    # @@ more complex?
#    Q = Graph(triples=[
#       # @@ check OWL properties
#       triple.Triple(cls, RDFS.subClassOf, VAR.R), 
#       triple.Triple(VAR.R, OWL.onProperty, prop), 
#       triple.Triple(VAR.R, OWL.cardinality, VAR.N)
#    ])
#    result = None
#    results = G.query(Q)
#    for result in results: 
#       result = int(result[VAR.N].value)
#    return result

def isOntology(G, term): 
   return triple.Triple(term, RDF.type, OWL.Ontology) in G

def isOWLClass(G, term): 
   pass # @@

def isRestriction(G, term): 
   return triple.Triple(term, RDF.type, OWL.Restriction) in G

def isDatatypeProperty(G, term): 
   return triple.Triple(term, RDF.type, OWL.DatatypeProperty) in G

def isObjectProperty(G, term): 
   return triple.Triple(term, RDF.type, OWL.ObjectProperty) in G

def isAnnotationProperty(G, term): 
   return triple.Triple(term, RDF.type, OWL.AnnotationProperty) in G

def isOntologyProperty(G, term): 
   return triple.Triple(term, RDF.type, OWL.OntologyProperty) in G

def isFunctionalProperty(G, term): 
   return triple.Triple(term, RDF.type, OWL.FunctionalProperty) in G

def isInverseFunctionalProperty(G, term): 
   return triple.Triple(term, RDF.type, OWL.InverseFunctionalProperty) in G

def isSymmetricProperty(G, term): 
   return triple.Triple(term, RDF.type, OWL.SymmetricProperty) in G

def isTransitiveProperty(G, term): 
   return triple.Triple(term, RDF.type, OWL.TransitiveProperty) in G

def isOWLProperty(G, term): 
   if isDatatypeProperty(G, term): return True
   if isObjectProperty(G, term): return True
   if isAnnotationProperty(G, term): return True
   if isOntologyProperty(G, term): return True
   if isInverseFunctionalProperty(G, term): return True
   if isSymmetricProperty(G, term): return True
   if isTransitiveProperty(G, term): return True
   else: return isProperty(G, term)

def isAnInverse(G, term): 
   Q = graph.Graph(triples=[triple.Triple(term, OWL.inverseOf, VAR.x)])
   return len(G.query(Q)) > 0

def getInverses(G, term): 
   Q = graph.Graph(triples=[triple.Triple(term, OWL.inverseOf, VAR.x)])
   return [binding[VAR.x] for binding in G.query(Q)]

def propertyHasCardinalityConstraint(G, term): 
   QX = graph.Graph(triples=[
      triple.Triple(VAR.restriction, OWL.onProperty, term), 
      triple.Triple(VAR.restriction, OWL.maxCardinality, VAR.x)
   ])
   if len(G.query(QX)) > 0: return True

   QY = graph.Graph(triples=[
      triple.Triple(VAR.restriction, OWL.onProperty, term), 
      triple.Triple(VAR.restriction, OWL.minCardinality, VAR.x)
   ])
   if len(G.query(QY)) > 0: return True

   QZ = Graph(triples=[
      triple.Triple(VAR.restriction, OWL.onProperty, term), 
      triple.Triple(VAR.restriction, OWL.cardinality, VAR.x)
   ])
   if len(G.query(QZ)) > 0: return True

   return False

def getCardinality(G, subject, predicate): 
   Q = graph.Graph(triples=[triple.Triple(subject, predicate, VAR.x)])
   return len(G.query(Q))

def importsClosure(G): 
   # @@ copy G?
   Q = graph.Graph(triples=[triple.Triple(VAR.x, OWL.imports, VAR.y)])

   uris = {}
   while 1: 
      foundURIs = False
      for uri in [binding[VAR.y)] for binding in G.query(Q)]: 
         if not uris.has_key(uri): 
            foundURIs = True
            uris[uri] = 0
            G += graph.Graph(uri=uri)
      if not foundURIs: break

   return G

if __name__=="__main__": 
   main()
