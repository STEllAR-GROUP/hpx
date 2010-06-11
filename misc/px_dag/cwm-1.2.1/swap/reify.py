#!/usr/bin/python
"""
Functions to reify and dereify a graph.
These functions should be perfect inverses of each other.

The strategy used is different from that of the reifier
in notation3.py, that tries to reify what it outputs.
This simply puts the reification into the sink given,
or a new one, depending on the function called.
$Id: reify.py,v 1.19 2007/12/11 21:18:08 syosi Exp $
"""
from term import BuiltIn, LightBuiltIn, LabelledNode, \
    HeavyBuiltIn, Function, ReverseFunction, AnonymousNode, \
    Literal, Symbol, Fragment, FragmentNil, Term,\
    CompoundTerm, List, EmptyList, NonEmptyList, N3Set
from formula import Formula, StoredStatement
from RDFSink import CONTEXT, PRED, SUBJ, OBJ, PARTS, ALL4, RDF_type_URI
import uripath
import diag
from mixin import Mixin
from set_importer import Set

reifyNS = 'http://www.w3.org/2004/06/rei#'
REIFY_NS = reifyNS
owlOneOf = 'http://www.w3.org/2002/07/owl#oneOf'

class rTerm(Mixin, Term):
    def unflatten(self, sink, bNodes, why=None):
        return self

class rLabelledNode(Mixin, LabelledNode):
    def reification(self, sink, bnodeMap={}, why=None):
        """Describe myself in RDF to the given context
        
        [ reify:uri "http://example.org/whatever"]
        """ #"
        b = sink.newBlankNode(why=why)
        uri = sink.newSymbol(REIFY_NS + "uri")
        sink.add(subj=b, pred=uri, obj=sink.newLiteral(self.uriref()), why=why)
        return b

    def flatten(self, sink, why=None):
        return self

class rAnonymousNode(Mixin, AnonymousNode):
    def reification(self, sink, bnodeMap={}, why=None):
        """Describe myself in RDF to the given context
        
        [ reify:items ( [ reify:value "foo"]  .... ) ]
        """
        try:
            return bnodeMap[self]
        except KeyError:
            b = sink.newBlankNode()
            bnodeMap[self] = b
        q = sink.newSymbol(RDF_type_URI)
        r = sink.newSymbol(REIFY_NS+"BlankNode")
        sink.add(subj=b, pred=q, obj=r, why=why)
        return b

    def flatten(self, sink, why=None):
        sink.declareExistential(self)
        return self

    def unflatten(self, sink, bNodes, why=None):
        try:
            return bNodes[self]
        except KeyError:
            bNodes[self] = sink.newBlankNode()
            return bNodes[self]

class rN3Set(Mixin, N3Set):
    def reification(self, sink, bnodeMap={}, why=None):
        """Describe myself in RDF to the given context
        
        [ reify:items ( [ reify:value "foo"]  .... ) ]
        """
        m = [x for x in self]
        m.sort(Term.compareAnyTerm)
        mm = [x.reification(sink, bnodeMap, why) for x in m]
        elements = sink.store.newSet(mm)
        b = sink.newBlankNode()
        sink.add(subj=b, pred=sink.newSymbol(REIFY_NS+"items"), obj=elements, why=why)
        return b

    def flatten(self, sink, why=None):
        newlist = sink.store.newSet([x.flatten(sink, why=why) for x in self])
        return newlist

    def unflatten(self, sink, bNodes, why=None):
        newlist = sink.store.newSet([x.unflatten(sink, bNodes, why=why) for x in self])
        return newlist


class rList(Mixin, List):
    def reification(self, sink, bnodeMap={}, why=None):
        """Describe myself in RDF to the given context
        
        [ reify:items ( [ reify:value "foo"]  .... ) ]
        """
        elements = sink.newList([ x.reification(sink, bnodeMap, why) for x in self])
        b = sink.newBlankNode()
        sink.add(subj=b, pred=sink.newSymbol(REIFY_NS+"items"), obj=elements, why=why)
        return b

    def flatten(self, sink, why=None):
        newlist = sink.newList([x.flatten(sink, why=why) for x in self])
        return newlist

    def unflatten(self, sink, bNodes, why=None):
        newlist = sink.newList([x.unflatten(sink, bNodes, why=why) for x in self])
        return newlist

class rLiteral(Mixin, Literal):
    def reification(self, sink, bnodeMap={}, why=None):
        """Describe myself in RDF to the given context
        
        [ reify:value "un expression quelconque"@fr ]
        """
        b = sink.newBlankNode(why=why)
        sink.add(subj=b, pred=sink.newSymbol(REIFY_NS+"value"), obj=self, why=why)
        return b

    def flatten(self, sink, why=None):
        return self


class rFormula(Mixin, Formula):
    def reification(self, sink, bnodeMap={}, why=None):
        """Describe myself in RDF to the given context
        
        
        """
        list = [].__class__
        try:
            return bnodeMap[self]
        except KeyError:
            F = sink.newBlankNode()
            bnodeMap[self] = F
        rei = sink.newSymbol(reifyNS[:-1])
        myMap = bnodeMap
        ooo = sink.newSymbol(owlOneOf)
        es = list(self.existentials())
        es.sort(Term.compareAnyTerm)
        us = list(self.universals())
        us.sort(Term.compareAnyTerm)
        for vars, vocab in ((es,  rei["existentials"]), 
                        (us, rei["universals"])):
            if diag.chatty_flag > 54:
                progress("vars=", vars)
                progress("vars=", [v.uriref() for v in vars])
            list = sink.store.nil.newList([x.reification(sink, myMap, why) for x in vars]) # sink.newLiteral(x.uriref())
            klass = sink.newBlankNode()
            sink.add(klass, ooo, list)
            sink.add(F, vocab, klass, why) 


        #The great list of statements
        statementList = []
        for s in self.statements:
            subj = sink.newBlankNode()
            sink.add(subj, rei["subject"], s[SUBJ].reification(sink, myMap, why), why) 
            sink.add(subj, rei["predicate"], s[PRED].reification(sink, myMap, why), why )
            sink.add(subj, rei["object"], s[OBJ].reification(sink, myMap, why), why) 
            statementList.append(subj)
            
    #The great class of statements
        StatementClass = sink.newBlankNode()
        realStatementList = sink.store.nil.newList(statementList)
        sink.add(StatementClass, ooo, realStatementList,  why)
    #We now know something!
        sink.add(F, rei["statements"], StatementClass, why)
            
        return F

    def flatten(self, sink, why=None):
        return self.reification(sink, {}, why=why)

def flatten(formula):
    """Flatten a formula

    This will minimally change a formula to make it valid RDF
    flattening a flattened formula should thus be the unity
    """
    why = None
    store = formula.store
    valid_triples = formula.statements[:]
    for triple in valid_triples:
        for part in SUBJ, PRED, OBJ:
            try:
                triple[part] = triple[part].close()
            except AttributeError:
                pass
    
    invalid_triples = []
    new_invalid_triples  = []
    shared_vars = Set()
    for triple in valid_triples:
        if triple.occurringIn(formula.universals()):
                new_invalid_triples.append(triple)
                shared_vars.update(triple.occurringIn(formula.existentials()))
    for triple in new_invalid_triples:
        try:
            valid_triples.remove(triple)
        except ValueError:
            pass
    while new_invalid_triples:
        invalid_triples.extend(new_invalid_triples)
        new_invalid_triples = []
        for triple in valid_triples:
            if triple.occurringIn(shared_vars):
                new_invalid_triples.append(triple)
                shared_vars.update(triple.occurringIn(formula.existentials()))
        for triple in new_invalid_triples:
            try:
                valid_triples.remove(triple)
            except ValueError:
                pass        
    still_needed_existentials = reduce(Set.union,
                                       [x.occurringIn(formula.existentials()) for x in valid_triples],
                                       Set())
    returnFormula = formula.newFormula()
    tempBindings = {}
    bNodeBindings = {}
    for a in still_needed_existentials:
        bNodeBindings = returnFormula.newBlankNode(a)
    tempformula = formula.newFormula()
    for var in formula.universals():
        tempBindings[var] = tempformula.newUniversal(var)
    for var in formula.existentials():
        termBindings[var] = tempformula.newBlankNode(var)
    for triple in invalid_triples:
        tempformula.add(triple[SUBJ].substitution(tempBindings, why=why),
                        triple[PRED].substitution(tempBindings, why=why),
                        triple[OBJ].substitution(tempBindings, why=why))
    #now for the stuff that isn't reified
    for triple in valid_triples:
        returnFormula.add(triple[SUBJ].substitution(bNodeBindings, why=why).flatten(returnFormula),
                        triple[PRED].substitution(bNodeBindings, why=why).flatten(returnFormula),
                        triple[OBJ].substitution(bNodeBindings, why=why).flatten(returnFormula))
    if tempformula.statements != []:
        x = tempformula.reification(returnFormula)
        returnFormula.add(x, store.type, store.Truth)
    return returnFormula.close()


def reify(formula):
    """Reify a formula

    Returns an RDF formula with the same semantics
    as the given formula
    """ 
    a = formula.newFormula()
    x = formula.reification(a)
    a.add(x, a.store.type, a.store.Truth)
    a = a.close()
    return a



def unflatten(formula, sink=None):
    """Undo the effects of the flatten function.

    Note that this still requires helper methods scattered throughout the
    Term heriarchy. 
    
    Our life is made much more difficult by the necessity of removing all
    triples that have been dereified --- this required a modification to dereification()

    """
    store = formula.store
    if sink == None:
        sink = formula.newFormula()
    bNodes = {}     #to track bNodes that bacome formulae
    rei = formula.newSymbol(reifyNS[:-1])
    formulaList = formula.each(pred=rei["statements"])

    #on the next line, we are filtering Truths because those will be included later
    formulaList = [a for a in formulaList if formula.the(pred=store.type, obj=store.Truth) == None]
    skipNodeList=[]
    for a in formulaList:
        xList = []
        bNodes[a] = dereification(a, formula, sink, xList=xList)
        skipNodeList.extend(xList[1:])
    dereify(formula, sink, xList=skipNodeList)
    for triple in formula.statements:
        if triple[PRED] != rei["statements"] and \
           triple[PRED] != rei["universals"] and \
           triple[PRED] != rei["value"] and \
           triple[PRED] != rei["existentials"] and \
           triple[SUBJ] not in skipNodeList and \
           triple[PRED] not in skipNodeList and \
           triple[OBJ] not in skipNodeList:
            sink.add(triple[SUBJ].unflatten(sink, bNodes),
                     triple[PRED].unflatten(sink, bNodes),
                     triple[OBJ].unflatten(sink, bNodes))        
    return sink.close()

#### Alternative method
# Shortcuts are too messy and don't work with lists
#
def dereify(formula, sink=None, xList=[]):
    store = formula.store
    if sink == None:
        sink = formula.newFormula()
    weKnowList = formula.each(pred=store.type, obj=store.Truth)
    for weKnow in weKnowList:
        f = dereification(weKnow, formula, sink, xList=xList)
        sink.loadFormulaWithSubstitution(f)
    return sink

def dereification(x, f, sink, bnodes={}, xList=[]):
    rei = f.newSymbol(reifyNS[:-1])
    xList.append(x)
    
    if x == None:
        raise ValueError, "Can't dereify nothing. Suspect missing information in reified form."
    y = f.the(subj=x, pred=rei["uri"])
    if y != None: return sink.newSymbol(y.value())
        
    y = f.the(subj=x, pred=rei["value"])
    if y != None: return y
    
    y = f.the(subj=x, pred=rei["items"])
    if y != None:
        if isinstance(y, N3Set):
            yy = [z for z in y]
            yy.sort(Term.compareAnyTerm)
            return sink.store.newSet([dereification(z, f, sink, bnodes, xList) for z in yy])
        return sink.newList([dereification(z, f, sink, bnodes, xList) for z in y])
    
    y = f.the(subj=x, pred=rei["statements"])
    if y != None:
        z = sink.newFormula()
        zbNodes = {}  # Bnode map for this formula
        
        uset = f.the(subj=x, pred=rei["universals"])
        xList.append(uset)
        ulist = uset #f.the(subj=uset, pred=f.newSymbol(owlOneOf))
        xList.append(ulist)
        from diag import progress
        if diag.chatty_flag > 54:
            progress("universals = ",ulist)
        for v in ulist:
            z.declareUniversal(f.newSymbol(v.value()))

        uset = f.the(subj=x, pred=rei["existentials"])
        xList.append(uset)
        ulist = uset #f.the(subj=uset, pred=f.newSymbol(owlOneOf))
        xList.append(ulist)
        if diag.chatty_flag > 54:
            progress("existentials %s =  %s"%(ulist, ulist.value()))
        for v in ulist:
            if diag.chatty_flag > 54:
                progress("Variable is ", v)
            z.declareExistential(f.newSymbol(v.value()))
        yy = y #f.the(subj=y, pred=f.newSymbol(owlOneOf))
        xList.append(yy)
        if diag.chatty_flag > 54:
            progress("Statements:  set=%s, list=%s = %s" %(y,yy, yy.value()))
        for stmt in yy:
            z.add(dereification(f.the(subj=stmt, pred=rei["subject"]), f, sink, zbNodes, xList),
                dereification(f.the(subj=stmt, pred=rei["predicate"]), f, sink, zbNodes, xList),
                dereification(f.the(subj=stmt, pred=rei["object"]), f, sink, zbNodes, xList))
        return z.close()
    y = f.the(subj=x, pred=f.newSymbol(RDF_type_URI))
    if y is not None:
        if x in bnodes:
            return bnodes[x]
        z = sink.newBlankNode()
        bnodes[x] = z
        return z
    
    raise ValueError, "Can't dereify %s - no clues I understand in %s" % (x, f)
