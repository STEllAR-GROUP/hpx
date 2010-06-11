#! /usr/bin/python
"""
    A module to allow one to chose which RDF parser to use.
    This is not done correctly, so I don't expect this file
    to last very long.

    $Id: rdfxml.py,v 1.3 2005/10/24 16:58:38 timbl Exp $

"""

def rdfxmlparser(store, openFormula, thisDoc=None,  flags="", why=None,
		    parser='sax2rdf'):
    if parser == 'rdflib':
        import rdflib_user
        return rdflib_user.rdflib_handoff(store, openFormula,thisDoc, why=why)
    else:   # parser == sax2xml
        import sax2rdf
        return  sax2rdf.RDFXMLParser(store, openFormula,  thisDoc=thisDoc,
			flags=flags, why=why)
