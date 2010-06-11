#!/usr/bin/env python
"""
Builtins for doing SPARQL queries in CWM

$Id: cwm_sparql.py,v 1.22 2007/11/18 02:01:56 syosi Exp $

"""

from swap.term import LightBuiltIn, Function, ReverseFunction, MultipleFunction,\
    MultipleReverseFunction, typeMap, LabelledNode, \
    CompoundTerm, N3Set, List, EmptyList, NonEmptyList, \
    Symbol, Fragment, Literal, Term, AnonymousNode, HeavyBuiltIn, toBool
import diag
progress = diag.progress

from RDFSink import RDFSink
from set_importer import Set

import uripath

from toXML import XMLWriter

try:
    from decimal import Decimal
except ImportError:
    from local_decimal import Decimal

from term import ErrorFlag as MyError

SPARQL_NS = 'http://www.w3.org/2000/10/swap/sparqlCwm'



class BI_truthValue(LightBuiltIn):
    def eval(self, subj, obj, queue, bindings, proof, query):
        if isinstance(subj, Literal):
##            print '%s makes %s' % (subj, toBool(str(subj), subj.datatype.fragid))
##            print '%s makes %s' % (obj, toBool(str(obj), obj.datatype.fragid))
##            print 'I got here on %s, %s, returning %s' % (subj, obj, toBool(str(subj), subj.datatype.fragid) is toBool(str(obj), obj.datatype.fragid))
            return toBool(str(subj), subj.datatype) is toBool(str(obj), obj.datatype)
        raise TypeError("%s type cannot be converted to boolean" % `subj.__class`)

class BI_typeErrorIsTrue(LightBuiltIn):
    """
Subject is anything (must be bound. 1 works well)
Object is a formula containing the test as its only triple

    """
    def eval(self, subj, obj, queue, bindings, proof, query):
        if len(obj) != 1:
            raise TypeError
        statement = obj.statements[0]
        try:
            return statement.predicate().eval(statement.subject(), statement.object(), queue, bindings, proof, query)
        except:
            return True

class BI_typeErrorReturner(LightBuiltIn, Function):
    def evalObj(self, subj, queue, bindings, proof, query):
        if len(subj) != 1:
            raise TypeError
        statement = subj.statements[0]
        try:
            return statement.predicate().evalObj(statement.subject(), queue, bindings, proof, query)
        except:
            return MyError()

class BI_equals(LightBuiltIn, Function, ReverseFunction):
    def eval(self, subj, obj, queue, bindings, proof, query):
        xsd = self.store.integer.resource
        if isinstance(subj, Symbol) and isinstance(obj, Symbol):
            return subj is obj
        if isinstance(subj, Fragment) and isinstance(obj, Fragment):
            return subj is obj
        if isinstance(subj, Literal) and isinstance(obj, Literal):
            if subj.datatype == xsd['boolean'] or obj.datatype == xsd['boolean']:
                return (toBool(str(subj), subj.datatype.resource is xsd and subj.datatype.fragid or None) ==
                        toBool(str(obj), obj.datatype.resource is xsd and obj.datatype.fragid or None))
            if not subj.datatype and not obj.datatype:
                return str(subj) == str(obj)
            if subj.datatype.fragid in typeMap and obj.datatype.fragid in typeMap:
                return subj.value() == obj.value()
            if subj.datatype != obj.datatype:
                raise TypeError(subj, obj)
            return str(subj) == str(obj)
        raise TypeError(subj, obj)
                
        

    def evalSubj(self, obj, queue, bindings, proof, query):
        return obj

    def evalObj(self,subj, queue, bindings, proof, query):
        return subj


class BI_lessThan(LightBuiltIn):
    def evaluate(self, subject, object):
        return (subject < object)

class BI_greaterThan(LightBuiltIn):
    def evaluate(self, subject, object):
        return (subject > object)

class BI_notGreaterThan(LightBuiltIn):
    def evaluate(self, subject, object):
        return (subject <= object)

class BI_notLessThan(LightBuiltIn):
    def evaluate(self, subject, object):
        return (subject >= object)


class BI_notEquals(LightBuiltIn):
    def eval(self, subj, obj, queue, bindings, proof, query):
        return not self.store.newSymbol(SPARQL_NS)['equals'].eval(subj, obj, queue, bindings, proof, query)

class BI_dtLit(LightBuiltIn, Function, ReverseFunction):
    def evalObj(self,subj, queue, bindings, proof, query):
        subj = [a for a in subj]
        if len(subj) != 2:
            raise ValueError
        subject, datatype = subj
        if not isinstance(subj, Literal) or not isinstance(datatype, LabelledNode):
            raise TypeError
        if subj.datatype:
            raise TypeError('%s must not have a type already' % subj)
        return self.store.newLiteral(str(subj), dt=datatype)

    def evalSubj(self, obj, queue, bindings, proof, query):
        if not isinstance(obj, Literal):
            raise TypeError('I can only find the datatype of a Literal, not a %s' % `obj.__class__.__name__`)
        return self.store.newList([self.store.newLiteral(str(obj)), obj.datatype])

class BI_langLit(LightBuiltIn, Function, ReverseFunction):
    def evalObj(self,subj, queue, bindings, proof, query):
        subj = [a for a in subj]
        if len(subj) != 2:
            raise ValueError
        subject, lang = subj
        if not isinstance(subj, Literal) or not isinstance(lang, Literal):
            raise TypeError
        if not lang:
            lang = None
        else:
            lang = str(lang)
        if subj.lang:
            raise TypeError('%s must not have a lang already' % subj)
        return self.store.newLiteral(str(subj), lang=lang)

    def evalSubj(self, obj, queue, bindings, proof, query):
        if not isinstance(obj, Literal):
            raise TypeError('I can only find the datatype of a Literal, not a %s' % `obj.__class__.__name__`)
        lang = obj.lang
        if not obj.lang:
            lang = ''
        return self.store.newList([self.store.newLiteral(str(obj)), self.store.newLiteral(lang)])


class BI_lamePred(HeavyBuiltIn, MultipleReverseFunction):
    def eval(self, subj, obj, queue, bindings, proof, query):
        return True
    def evalSubj(self, obj, queue, bindings, proof, query):
        #really slow. Return every list I know anything about. Should I do this at all?
        retVals = Set([self.store.nil])
        retValCopy = Set()
        n = 0
        while retVals != retValCopy:
            print n, retVals, retValCopy
            n += 1
            retValCopy = retVals.copy()
            for node in retValCopy:
                retVals.update(node._prec.values())
        a = query.workingContext.occurringIn(retVals) ## Really slow. Need to generate this on the fly?
        print 'a=', a
        return a

#############################
#############################
#    Builtins useful from within cwm, not within SPARQL
#
#############################
#############################

    

class BI_query(LightBuiltIn, Function):
    def evalObj(self,subj, queue, bindings, proof, query):
        from query import applySparqlQueries
        ns = self.store.newSymbol(SPARQL_NS)
        assert isinstance(subj, List)
        subj = [a for a in subj]
        assert len(subj) == 2
        source, query = subj
        F = self.store.newFormula()
        applySparqlQueries(source, query, F)
        if query.contains(obj=ns['ConstructQuery']):
            return F
        if query.contains(obj=ns['SelectQuery']) or query.contains(obj=ns['AskQuery']):
            return self.store.newLiteral(sparql_output(query, F))

def bnode_replace(self, string):
    if string in self:
        return self[string]
    hash = string.find('#')
    base = string[:hash]
    self[string] = base + '#_bnode_' + str(self['counter'])
    self['counter'] += 1
    return self[string]

bnode_replace = bnode_replace.__get__({'counter':0})


def sparql_output(query, F):
    store = F.store
    RESULTS_NS = 'http://www.w3.org/2005/sparql-results#'
    ns = store.newSymbol(SPARQL_NS)
    if query.contains(obj=ns['SelectQuery']):
        node = query.the(pred=store.type, obj=ns['SelectQuery'])
        outputList = []
        prefixTracker = RDFSink()
        prefixTracker.setDefaultNamespace(RESULTS_NS)
        prefixTracker.bind('', RESULTS_NS)
        xwr = XMLWriter(outputList.append, prefixTracker)
        xwr.makePI('xml version="%s"' % '1.0')
        xwr.startElement(RESULTS_NS+'sparql', [], prefixTracker.prefixes)
        xwr.startElement(RESULTS_NS+'head', [], prefixTracker.prefixes)
        vars = []
        for triple in query.the(subj=node, pred=ns['select']):
            vars.append(triple.object())
            xwr.emptyElement(RESULTS_NS+'variable', [(RESULTS_NS+' name', str(triple.object()))], prefixTracker.prefixes)

        xwr.endElement()
        xwr.startElement(RESULTS_NS+'results', [], prefixTracker.prefixes)
        resultFormulae = [aa for aa in F.each(pred=store.type, obj=ns['Result'])]
        try:
            resultFormulae.sort(Term.compareAnyTerm)
        except:
            print [type(x) for x in resultFormulae]
            print Term
            raise
        for resultFormula in resultFormulae:
            xwr.startElement(RESULTS_NS+'result', [], prefixTracker.prefixes)
            for var in vars:
                binding = resultFormula.the(pred=ns['bound'], obj=var)
                if binding:
                    xwr.startElement(RESULTS_NS+'binding', [(RESULTS_NS+' name', str(var))],  prefixTracker.prefixes)
                    if isinstance(binding, LabelledNode):
                        xwr.startElement(RESULTS_NS+'uri', [],  prefixTracker.prefixes)
                        xwr.data(binding.uriref())
                        xwr.endElement()
                    elif isinstance(binding, (AnonymousNode, List)):
                        xwr.startElement(RESULTS_NS+'bnode', [],  prefixTracker.prefixes)
                        xwr.data(bnode_replace(binding.uriref()))
                        xwr.endElement()
                    elif isinstance(binding, Literal):
                        props = []
                        if binding.datatype:
                            props.append((RESULTS_NS+' datatype', binding.datatype.uriref()))
                        if binding.lang:
                            props.append(("http://www.w3.org/XML/1998/namespace lang", binding.lang))
                        xwr.startElement(RESULTS_NS+'literal', props,  prefixTracker.prefixes)
                        xwr.data(unicode(binding))
                        xwr.endElement()
                    xwr.endElement()
                else:
                    pass

            xwr.endElement()
        xwr.endElement()
        xwr.endElement()
        xwr.endDocument()
        return u''.join(outputList)
    if query.contains(obj=ns['AskQuery']):
        node = query.the(pred=store.type, obj=ns['AskQuery'])
        outputList = []
        prefixTracker = RDFSink()
        prefixTracker.setDefaultNamespace(RESULTS_NS)
        prefixTracker.bind('', RESULTS_NS)
        xwr = XMLWriter(outputList.append, prefixTracker)
        xwr.makePI('xml version="%s"' % '1.0')
        xwr.startElement(RESULTS_NS+'sparql', [], prefixTracker.prefixes)
        xwr.startElement(RESULTS_NS+'head', [], prefixTracker.prefixes)
        vars = []
#            for triple in query.the(subj=node, pred=ns['select']):
#                vars.append(triple.object())
#                xwr.emptyElement(RESULTS_NS+'variable', [(RESULTS_NS+'name', str(triple.object()))], prefixTracker.prefixes)

        xwr.endElement()
        xwr.startElement(RESULTS_NS+'boolean', [], prefixTracker.prefixes)
        if F.the(pred=store.type, obj=ns['Success']):
            xwr.data('true')
        else:
            xwr.data('false')
        xwr.endElement()
            

        xwr.endElement()
        xwr.endDocument()
        return ''.join(outputList)


def sparql_queryString(source, queryString):
    from query import applySparqlQueries
    store = source.store
    ns = store.newSymbol(SPARQL_NS)
    from sparql import sparql_parser
    import sparql2cwm
    convertor = sparql2cwm.FromSparql(store)
    import StringIO
    p = sparql_parser.N3Parser(StringIO.StringIO(queryString), sparql_parser.branches, convertor)
    q = p.parse(sparql_parser.start).close()
    F = store.newFormula()
    applySparqlQueries(source, q, F)
    F = F.close()
##    print 'result is ', F
##    print 'query is ', q.n3String()
    return outputString(q, F)
    
def outputString(q, F):
    store = q.store
    ns = store.newSymbol(SPARQL_NS)
    if q.contains(obj=ns['ConstructQuery']):
        return F.rdfString().decode('utf_8'), 'application/rdf+xml'
    if q.contains(obj=ns['SelectQuery']) or q.contains(obj=ns['AskQuery']):
        return sparql_output(q, F), 'application/sparql-results+xml'


class BI_semantics(HeavyBuiltIn, Function):
    """ The semantics of a resource are its machine-readable meaning, as an
    N3 forumula.  The URI is used to find a represnetation of the resource in bits
    which is then parsed according to its content type."""
    def evalObj(self, subj, queue, bindings, proof, query):
        store = subj.store
        if isinstance(subj, Fragment): doc = subj.resource
        else: doc = subj
        F = store.any((store._experience, store.semantics, doc, None))
        if F != None:
            if diag.chatty_flag > 10: progress("Already read and parsed "+`doc`+" to "+ `F`)
            return F

        if diag.chatty_flag > 10: progress("Reading and parsing " + doc.uriref())
        inputURI = doc.uriref()
        F = self.store.load(inputURI, contentType="x-application/sparql")
        if diag.chatty_flag>10: progress("    semantics: %s" % (F))
        if diag.tracking:
            proof.append(F.collector)
        return F.canonicalize()

def register(store):
    ns = store.newSymbol(SPARQL_NS)
    ns.internFrag('equals', BI_equals)
    ns.internFrag('lessThan', BI_lessThan)
    ns.internFrag('greaterThan', BI_greaterThan)
    ns.internFrag('notGreaterThan', BI_notGreaterThan)
    ns.internFrag('notLessThan', BI_notLessThan)
    ns.internFrag('notEquals', BI_notEquals)
    ns.internFrag('typeErrorIsTrue', BI_typeErrorIsTrue)
    ns.internFrag('typeErrorReturner', BI_typeErrorReturner)
    ns.internFrag('truthValue', BI_truthValue)
    ns.internFrag('lamePred', BI_lamePred)
    ns.internFrag('query', BI_query)
    ns.internFrag('semantics', BI_semantics)
    ns.internFrag('dtLit', BI_dtLit)
    ns.internFrag('langLit', BI_langLit)
