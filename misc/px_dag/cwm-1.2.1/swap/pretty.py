#! /usr/bin/python
"""

$Id: pretty.py,v 1.41 2007/09/09 22:51:42 timbl Exp $

Printing of N3 and RDF formulae

20003-8-20 split offf from llyn.py

This is or was http://www.w3.org/2000/10/swap/pretty.py
"""


import types
import string

import diag
from diag import progress, verbosity, tracking
from term import   Literal, XMLLiteral, Symbol, Fragment, AnonymousNode, \
    AnonymousVariable, FragmentNil, AnonymousUniversal, \
    Term, CompoundTerm, List, EmptyList, NonEmptyList, N3Set
from formula import Formula, StoredStatement

from RDFSink import Logic_NS, RDFSink, forSomeSym, forAllSym
from RDFSink import CONTEXT, PRED, SUBJ, OBJ, PARTS, ALL4, \
            ANONYMOUS, SYMBOL, LITERAL, LITERAL_DT, LITERAL_LANG, XMLLITERAL
from RDFSink import N3_nil, N3_first, N3_rest, OWL_NS, N3_Empty, N3_List, \
                    List_NS
from RDFSink import RDF_NS_URI
from RDFSink import RDF_type_URI

cvsRevision = "$Revision: 1.41 $"

# Magic resources we know about

from RDFSink import RDF_type_URI, DAML_sameAs_URI

STRING_NS_URI = "http://www.w3.org/2000/10/swap/string#"
META_NS_URI = "http://www.w3.org/2000/10/swap/meta#"
INTEGER_DATATYPE = "http://www.w3.org/2001/XMLSchema#integer"
FLOAT_DATATYPE = "http://www.w3.org/2001/XMLSchema#double"

prefixchars = "abcdefghijklmnopqustuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"



# Utility functions for the object-free abstract notation interface

def auPair(x):
    """Encode as object-free form for unparser interface"""
    if isinstance(x, XMLLiteral):
        return (XMLLITERAL, x.dom)
    if isinstance(x, Literal):
        if x.datatype:
            return LITERAL_DT, (str(x), x.datatype.uriref()) # could be XMLLit
        if x.lang:
            return LITERAL_LANG, (x.string, x.lang)
        return (LITERAL, x.string)

    if isinstance(x, AnonymousUniversal):
        if x.uri:
            return (SYMBOL, x.uriref())
    if isinstance(x, AnonymousNode):
        return (ANONYMOUS, x.uriref())
    else:
        return (SYMBOL, x.uriref())


def auxPairs(t):
    return(auPair(t[CONTEXT]),
                        auPair(t[PRED]),
                        auPair(t[SUBJ]),
                        auPair(t[OBJ]),
                        )


class TooMuchRecursion(Exception): pass

class Serializer:
    """A serializer to serialize the formula F into the given
    abstract syntax sink
    """
    def __init__(self, F, sink, flags="", sorting=0):
        self.context = F
#       assert F.canonical is not None, "Formula printed must be canonical"
        self.store = F.store
        self.sink = sink
        self.defaultNamespace = None
        self.flags = flags
        self.sorting = sorting
        self._inContext ={}
        self._loopCheck = {}
        self._inLoop = {}
        self._tooDeep = {}
        self._occurringAs = [{}, {}, {}, {}]
        self._topology_returns = {}
        
    def selectDefaultPrefix(self, printFunction):

        """ Symbol whose fragments have the most occurrences.
        we suppress the RDF namespace itself because the XML syntax
        has problems with it being default as it is used for attributes.
        
        This also outputs the prefixes."""

        if "d" in self.flags:
            self.defaultNamespace = None
            self.dumpPrefixes()
            return

        dummySink = self.sink.dummyClone()
        dummySerializer = Serializer(self.context,
            sink=dummySink, flags=self.flags+"d", sorting=self.sorting)
        printFunction(dummySerializer)


        best = 0
        mp = None
        counts = dummySink.namespaceCounts()
        for r, count in counts.items():
            if verbosity() > 25: progress("    Count is %3i for %s" %(count, r))
            if (r != RDF_NS_URI
                and count > 0
                and (count > best or
                     (count == best and mp > r))) :  # Must be repeatable for retests
                best = count
                mp = r

        if verbosity() > 20:
            progress("# Most popular Namespace is %s with %i" % ( mp, best))

        self.defaultNamespace = mp

        # Make up prefixes for things which don't have them:
        
        for r, count in counts.items():
            if count > 1 and r != mp:
                if self.store.prefixes.get(r, None) is None:
                    p = r
                    if p[-1] in "/#": p = p[:-1]
                    slash = p.rfind("/")
                    if slash >= 0: p = p[slash+1:]
                    i = 0
                    while i < len(p):
                        if p[i] in prefixchars:
                            i = i + 1
                        else:
                            break
                    p = p[:i]
                    if len(p) <6 and self.store.namespaces.get(p, None)is None: # and p[:3]!='xml':
                        pref = p
                    else:
                        p = p[:5]
                        for l in (3, 2, 4, 1, 5):
                            if self.store.namespaces.get(p[:l], None) is None: # and p[:l][:3]!='xml':
                                pref = p[:l]
                                break   
                        else:
                            n = 2
                            while 1:
                                pref = p[:3]+`n`
                                if self.store.namespaces.get(pref, None) is None:
                                    break
                                n = n + 1                       
    
                    self.store.bind(pref, r)
                    if verbosity() > 50: progress("Generated @prefix %s: <%s>." % (pref, r))

        if self.defaultNamespace is not None:
            self.sink.setDefaultNamespace(self.defaultNamespace)

#       progress("&&&& Counts: ", counts)
        prefixes = self.store.namespaces.keys()   #  bind in same way as input did FYI
        prefixes.sort()   # For repeatability of test results
        for pfx in prefixes:
            r = self.store.namespaces[pfx]
            try:
                count = counts[r]
                if count > 0:
                    self.sink.bind(pfx, r)
            except KeyError:
                pass
        return

    def dumpPrefixes(self):
        if self.defaultNamespace is not None:
            sink.setDefaultNamespace(self.defaultNamespace)
        prefixes = self.store.namespaces.keys()   #  bind in same way as input did FYI
        prefixes.sort()
        for pfx in prefixes:
            uri = self.store.namespaces[pfx]
            self.sink.bind(pfx, uri)


    def _listsWithinLists(self, L, lists):
        if L not in lists:
            lists.append(L)
        for i in L:
            if isinstance(i, NonEmptyList) or isinstance(i, N3Set):
                self._listsWithinLists(i, lists)

    def dumpLists(self):
        "Dump lists out as first and rest. Not used in pretty."
        listList = {}
        context = self.context
        sink = self.sink
        lists = []
        for s in context.statements:

            for x in s.predicate(), s.subject(), s.object():
                if isinstance(x, NonEmptyList) or isinstance(x, N3Set):
                    self._listsWithinLists(x, lists)
                    
        for l in lists:
            if isinstance(l, N3Set):
                a = context.newBlankNode()
                ll = [mm for mm in l] #I hate sorting things
                ll.sort(Term.compareAnyTerm)
                list = self.store.newList(ll)
                self._outputStatement(sink, (context, self.store.forSome, context, a))
                l._list = list
                l._node = a
            else:
                list = l
            while not isinstance(list, EmptyList):
                if list not in listList:
                    self._outputStatement(sink, (context, self.store.forSome, context, list))
                    listList[list] = 1
                list = list.rest
        listList = {}
        for l in lists:
            list = l
            if isinstance(l, N3Set):
                list = l._list
                self._outputStatement(sink, (context, self.store.owlOneOf, l._node, list))
            while (not isinstance(list, EmptyList)) and list not in listList:
                if isinstance(list.first, N3Set):
                    self._outputStatement(sink, (context, self.store.first, list, list.first._node))
                else:
                    self._outputStatement(sink, (context, self.store.first, list, list.first))
                self._outputStatement(sink, (context, self.store.rest,  list, list.rest))
                listList[list] = 1
                list = list.rest


        

    def dumpChronological(self):
        "Fast as possible. Only dumps data. No formulae or universals."
        context = self.context
        sink = self.sink
        sink.startDoc()
        self.dumpPrefixes()
        self.dumpVariables(context, sink, sorting=0, dataOnly=1)
        uu = context.universals()

        self.dumpLists()

        def fixSet(x):
            try:
                return x._node
            except AttributeError:
                return x
            
        for s in context.statements:
            for p in SUBJ, PRED, OBJ:
                x = s[p]
                if isinstance(x, Formula) or x in uu:
                    break
            else:
                self._outputStatement(sink, [fixSet(x) for x in s.quad])
                    
        sink.endDoc()

    def _outputStatement(self, sink, quad, aWorks = 1):
        if isinstance(quad[1], Literal):
            raise ValueError("Cannot have a literal as a predicate. This makes no sense, %s" % `quad[1]`)
        if isinstance(quad[1], Formula):
            raise ValueError("Cannot have a formula as a predicate. This makes no sense")
        sink.makeStatement(auxPairs(quad), aIsPossible=aWorks)


    def dumpVariables(self, context, sink, sorting=1, pretty=0, dataOnly=0):
        """Dump the forAlls and the forSomes at the top of a formula"""
        uv = list(context.universals())
        ev = list(context.existentials())
        if sorting:
            uv.sort(Term.compareAnyTerm)
            ev.sort(Term.compareAnyTerm)
        if not dataOnly:
            for v in uv:
                self._outputStatement(sink, (context, self.store.forAll, context, v))
        for v in ev:
            aWorks = 0
            if pretty:
                _anon, _incoming = self._topology(v, context)
                if not _anon:
                    self._outputStatement(sink, (context, self.store.forSome, context, v), \
                                        canItbeABNode(context, v))
            else: # not pretty, no formulae, can always use _:a form
                self._outputStatement(sink, (context, self.store.forSome, context, v), 1)

    def dumpBySubject(self, sorting=1):
        """ Dump one formula only by order of subject except
            forSome's first for n3=a mode"""
        
        context = self.context
        uu = context.universals().copy()
        sink = self.sink
        self._scan(context)
        sink.startDoc()
        self.selectDefaultPrefix(Serializer.dumpBySubject)        
        self.dumpVariables(context, sink, sorting)
        self.dumpLists()

        ss = context.statements[:]
        ss.sort()
        def fixSet(x):
            try:
                return x._node
            except AttributeError:
                return x
    
        for s in ss:
            for p in SUBJ, PRED, OBJ:
                x = s[p]
                if isinstance(x, Formula) or x in uu:
                    break
            else:
                self._outputStatement(sink, [fixSet(x) for x in s.quad])
                    
        if 0:  # Doesn't work as ther ei snow no list of bnodes
            rs = self.store.resources.values()
            if sorting: rs.sort(Term.compareAnyTerm)
            for r in rs :  # First the bare resource
                statements = context.statementsMatching(subj=r)
                if sorting: statements.sort(StoredStatement.comparePredObj)
                for s in statements :
                        self._outputStatement(sink, s.quad)
                if not isinstance(r, Literal):
                    fs = r.fragments.values()
                    if sorting: fs.sort
                    for f in fs :  # then anything in its namespace
                        statements = context.statementsMatching(subj=f)
                        if sorting: statements.sort(StoredStatement.comparePredObj)
                        for s in statements:
                            self._outputStatement(sink, s.quad)
        sink.endDoc()
#
#  Pretty printing
#
# An intersting alternative is to use the reverse syntax to the max, which
# makes the DLG an undirected labelled graph. s and o above merge. The only think which
# then prevents us from dumping the graph without new bnode ids is the presence of cycles.
#
# Blank nodes can be represented using the implicit syntax [] or rdf/xml equivalent
# instead of a dummy identifier iff
# - they are blank nodes, ie are existentials whose id has been generated, and
# - the node only occurs directly in one formula in the whole thing to be printed, and
# - the node occurs at most once as a object or list element within that formula

# We used to work this out on the fly, but it is faster to build an index of the
# whole formula to be printed first.
#
# Note when we scan a list we do it in the context of the formula in which we found
# it.  It may occcur in many formulae.

    def _scanObj(self, context, x):
        "Does this appear in just one context, and if so counts how many times as object"
        z = self._inContext.get(x, None)
        if z == "many": return # forget it
        if z is None:
            self._inContext[x] = context
        elif z is not context:
            self._inContext[x] = "many"
            return
        if isinstance(x, NonEmptyList) or isinstance(x, N3Set):
            for y in x:
                self._scanObj(context, y)           
        if isinstance(x, AnonymousVariable) or (isinstance(x, Fragment) and x.generated()): 
            y = self._occurringAs[OBJ].get(x, 0) + 1
            self._occurringAs[OBJ][x] = y
            if verbosity() > 98:
                progress(
                    "scan: %s, a %s, now has %i occurrences as %s" 
                    %(x, x.__class__,y,"CPSOqqqqq"[y]))
#       else:
#           if x is None: raise RuntimeError("Weird - None in a statement?")
#           progress("&&&&&&&&& %s has class %s " %(`z`, `z.__class__`))

    def _scan(self, x, context=None):
#       progress("Scanning ", x, " &&&&&&&&")
#       assert self.context._redirections.get(x, None) is None, "Should not be redirected: "+`x`
        if verbosity() > 98: progress("scanning %s a %s in context %s" %(`x`, `x.__class__`,`context`),
                        x.generated(), self._inContext.get(x, "--"))
        if isinstance(x, NonEmptyList) or isinstance(x, N3Set):
            for y in x:
                self._scanObj(context, y)
        if isinstance(x, Formula):
            for s in x.statements:
                for p in PRED, SUBJ, OBJ:
                    y = s[p]
                    if (isinstance(y, AnonymousVariable) 
                        or (isinstance(y, Fragment) and y.generated())): 
                        z = self._inContext.get(y, None)
                        if z == "many": continue # forget it
                        if z is None:
                            self._inContext[y] = x
                        elif z is not x:
                            self._inContext[y] = "many"
                            continue
                        z = self._occurringAs[p].get(y, 0)
                        self._occurringAs[p][y] = z + 1
#                       progress("&&&&&&&&& %s now occurs %i times as %s" %(`y`, z+1, "CPSO"[p]))
#                   else:
#                       progress("&&&&&&&&& yyyy  %s has class %s " %(`y`, `y.__class__`))
                    if x is not y: self._scan(y, x)
            self._breakloops(x)
                
    def _breakloops(self, context):
        _done = {}
        _todo = list(self._occurringAs[SUBJ])
        _todo.sort(Term.compareAnyTerm)
        for x in _todo:
            if x in _done:
                continue
            if not (isinstance(x, AnonymousVariable) and not ((isinstance(x, Fragment) and x.generated()))):
                _done[x] = True
                continue
            _done[x] = x
            a = True
            y = x
            while a is not None:
                a = context._index.get((None, None, y), None)
                if a is None or len(a) == 0:
                    a = None
                if a is not None:
                    #print a
                    y = a[0][SUBJ]
                    beenHere = _done.get(y, None)
                    if beenHere is x:
                        self._inLoop[y] = 1
                        a = None
                    elif beenHere is not None:
                        a = None
                    elif not (isinstance(y, AnonymousVariable) and not ((isinstance(y, Fragment) and y.generated()))):
                        _done[y] = True
                        a = None
                    else:
                        _done[y] = x

    def _topology(self, x, context): 
        """ Can be output as an anonymous node in N3. Also counts incoming links.
        Output tuple parts:

        1. True iff can be represented as anonymous node in N3, [] or {}
        2. Number of incoming links: >0 means occurs as object or pred, 0 means as only as subject.
            1 means just occurs once
            >1 means occurs too many times to be anon
        
        Returns  number of incoming links (1 or 2) including forSome link
        or zero if self can NOT be represented as an anonymous node.
        Paired with this is whether this is a subexpression.
        """
    # This function takes way too long. My attempts to speed it up using a try / except
    # loop were clearly misguided, because this function does very little as is.
    # why does this take .08 seconds per function call to do next to nothing?
##        try:
##            return self._topology_returns[x]
##        except KeyError:
##            pass
#       progress("&&&&&&&&& ", `self`,  self._occurringAs)
        _isExistential = x in context.existentials()
#        _isExistential = context.existentialDict.get(x,0)
#        return (0, 2)
        _loop = context.any(subj=x, obj=x)  # does'nt count as incomming
        _asPred = self._occurringAs[PRED].get(x, 0)
        _asObj = self._occurringAs[OBJ].get(x, 0)
        _inLoop = self._inLoop.get(x, 0)
        _tooDeep = self._tooDeep.get(x, 0)
        if isinstance(x, Literal):
            _anon = 0     #  Never anonymous, always just use the string
        elif isinstance(x, Formula):
            _anon = 2   # always anonymous, represented as itself
                
        elif isinstance(x, List):
            if isinstance(x, EmptyList):
                _anon = 0     #  Never anonymous, always just use the string
            else:
                _anon = 2       # always anonymous, represented as itself
                _isExistential = 1
        elif isinstance(x, N3Set):
            _anon = 2
            _isExistential = 1
        elif not x.generated():
            _anon = 0   # Got a name, not anonymous
        else:  # bnode
            ctx = self._inContext.get(x, "weird")
            _anon = ctx == "weird" or (ctx is context and
                        _asObj < 2 and _asPred == 0 and _inLoop == 0 and 
                        _tooDeep == 0 and (not _loop) and
                        _isExistential)
            if verbosity() > 97:
                progress( "Topology %s in %s is: ctx=%s,anon=%i obj=%i, pred=%i loop=%s ex=%i "%(
                `x`, `context`, `ctx`, _anon, _asObj, _asPred, _loop, _isExistential))
            return ( _anon, _asObj+_asPred )  

        if verbosity() > 98:
            progress( "Topology %s in %s is: anon=%i obj=%i, pred=%i loop=%s ex=%i "%(
            `x`, `context`,  _anon, _asObj, _asPred, _loop, _isExistential))
##        self._topology_returns[x] = ( _anon, _asObj+_asPred )
        return ( _anon, _asObj+_asPred )  


  
    def dumpNested(self):
        """ Iterates over all URIs ever seen looking for statements
        """

        context = self.context
#        assert context.canonical is not None
        self._scan(context)
        self.sink.startDoc()
        self.selectDefaultPrefix(Serializer.dumpNested)        
        self.dumpFormulaContents(context, self.sink, sorting=1, equals=1)
        self.sink.endDoc()

    def tmDumpNested(self):
        """

        """
        context = self.context
        assert context.canonical is not None
        self._scan(context)
        self.tm.start()
        self._dumpFormula(context)
        self.tm.end()

    def _dumpNode(self, node):
        tm = self.tm
        _anon, _incoming = self._topology(node, context)
        if isinstance(node, List):
            tm.startList()
            [self._dumpNode(x) for x in node]
            tm.endList()
        elif isinstance(node, N3Set):
            pass
        elif isinstance(node, formula):
            tm.startFormula()
            self._dumpFormula(node)
            tm.endFormula()
        elif _anon:
            tm.startAnonymous()
            self._dumpSubject(context, node)
            tm.endAnonymous()
        elif isinstance(node, Literal):
            tm.addLiteral(node)
        elif isinstance(node, Symbol):
            tm.addSymbol(node)
        else:
            pass
        
    def _dumpFormula(self, node):
        pass

    def _dumpSubject(self, formula, node):
        pass

    def _dumpPredicate(self, formula, subject, node):
        pass

    def dumpFormulaContents(self, context, sink, sorting, equals=0):
        """ Iterates over statements in formula, bunching them up into a set
        for each subject.
        """

        allStatements = context.statements[:]
        if equals:
            for x, y in context._redirections.items():
                if not x.generated() and x not in context.variables():
                    allStatements.append(StoredStatement(
                        (context, context.store.sameAs, x, y)))
        allStatements.sort()
#        context.statements.sort()
        # @@ necessary?
        self.dumpVariables(context, sink, sorting, pretty=1)

#       statements = context.statementsMatching(subj=context)  # context is subject
#        if statements:
#           progress("@@ Statement with context as subj?!", statements,)
#            self._dumpSubject(context, context, sink, sorting, statements)

        currentSubject = None
        statements = []
        for s in allStatements:
            con, pred, subj, obj =  s.quad
            if subj is con: continue # Done them above
            if currentSubject is None: currentSubject = subj
            if subj != currentSubject:
                self._dumpSubject(currentSubject, context, sink, sorting, statements)
                statements = []
                currentSubject = subj
            statements.append(s)
        if currentSubject is not None:
            self._dumpSubject(currentSubject, context, sink, sorting, statements)


    def _dumpSubject(self, subj, context, sink, sorting, statements=[]):
        """ Dump the infomation about one top level subject
        
        This outputs arcs leading away from a node, and where appropriate
     recursively descends the tree, by dumping the object nodes
     (and in the case of a compact list, the predicate (rest) node).
     It does NOTHING for anonymous nodes which don't occur explicitly as subjects.

     The list of statements must be sorted if sorting is true.     
        """
        _anon, _incoming = self._topology(subj, context)    # Is anonymous?
        if _anon and  _incoming == 1 and not isinstance(subj, Formula): return   # Forget it - will be dealt with in recursion

        if isinstance(subj, List): li = subj
        else: li = None
        if isinstance(subj, N3Set):
            #I hate having to sort things
            se = [mm for mm in subj]
            se.sort(Term.compareAnyTerm)
            li = self.store.newList(se)
        else: se = None
        
        if isinstance(subj, Formula) and subj is not context:
            sink.startFormulaSubject(auPair(subj))
            self.dumpFormulaContents(subj, sink, sorting) 
            sink.endFormulaSubject(auPair(subj))       # Subject is now set up
            # continue to do arcs
            
        elif _anon and (_incoming == 0 or 
            (li is not None and not isinstance(li, EmptyList))):    # Will be root anonymous node - {} or [] or ()
                
            if subj is context:
                pass
            else:     #  Could have alternative syntax here

                if sorting: statements.sort(StoredStatement.comparePredObj) # @@ Needed now Fs are canonical?

                if se is not None:
                    a = self.context.newBlankNode()
                    sink.startAnonymousNode(auPair(a))
                    self.dumpStatement(sink, (context, self.store.owlOneOf, a, li), sorting)
                    for s in statements:  #   "[] color blue."  might be nicer. @@@  Try it?
                        m = s.quad[0:2] + (a, s.quad[3])
                        self.dumpStatement(sink, m, sorting)
                    sink.endAnonymousNode()
                    return 
                elif li is not None and not isinstance(li, EmptyList):
                    for s in statements:
                        p = s.quad[PRED]
                        if p is not self.store.first and p is not self.store.rest:
                            if verbosity() > 90: progress("Is list, has values for", `p`)
                            break # Something to print (later)
                    else:
                        if subj.generated(): return # Nothing.

                    if "l" not in self.flags:
                        sink.startListSubject(auPair(subj))
                        for ele in subj:
                            self.dumpStatement(sink, (context, self.store.li, subj,
                                                ele), sorting)
                        sink.endListSubject(auPair(subj))
                        for s in statements:
                            p = s.quad[PRED]
                            if p is not self.store.first and p is not self.store.rest:
                                self.dumpStatement(sink, s.quad, sorting) # Dump the rest outside the ()
                    else:  # List but explicitly using first and rest
                        sink.startAnonymousNode(auPair(subj))
                        self.dumpStatement(sink, (context,
                            self.store.first, subj, subj.first), sorting)
                        self.dumpStatement(sink, (context,
                            self.store.rest, subj, subj.rest), sorting)
                        for s in statements:
                                self.dumpStatement(sink, s.quad, sorting)
                        sink.endAnonymousNode()
                    return
                else:
                    if verbosity() > 90: progress("%s Not list, has property values." % `subj`)
                    sink.startAnonymousNode(auPair(subj))
                    for s in statements:  #   "[] color blue."  might be nicer. @@@  Try it?
                        try:
                            self.dumpStatement(sink, s.quad, sorting)
                        except TooMuchRecursion:
                            pass
                    sink.endAnonymousNode()
                    return  # arcs as subject done


        if sorting: statements.sort(StoredStatement.comparePredObj)
        for s in statements:
            self.dumpStatement(sink, s.quad, sorting)

                
    def dumpStatement(self, sink, triple, sorting):
        "Dump one statement, including structure within object" 

        context, pre, sub, obj = triple
        if (sub is obj and not isinstance(sub, CompoundTerm))  \
           or (isinstance(obj, EmptyList)) \
           or isinstance(obj, Literal):
            self._outputStatement(sink, triple) # Do 1-loops simply
            return


        if isinstance(obj, Formula):
            sink.startFormulaObject(auxPairs(triple))
            self.dumpFormulaContents(obj, sink, sorting)
            sink.endFormulaObject(auPair(pre), auPair(sub))
            return

        if isinstance(obj, NonEmptyList):
            if verbosity()>99:
                progress("List found as object of dumpStatement " + `obj`
                                        + context.debugString())

            collectionSyntaxOK = ("l" not in self.flags)
#           if "x" in self.flags:  #  Xml can't serialize literal in collection
#               for ele in obj:
#                   if isinstance(ele, Literal):
#                       collectionSyntaxOK = 0
#                       break

            if collectionSyntaxOK:
                sink.startListObject(auxPairs(triple))
                for ele in obj:
                    self.dumpStatement(sink, (context, self.store.li, obj, ele),
                        sorting)
                sink.endListObject(auPair(sub), auPair(pre))
            else:
                sink.startAnonymous(auxPairs(triple))
                self.dumpStatement(sink,
                    (context, self.store.first, obj, obj.first), sorting)
                self.dumpStatement(sink,
                    (context, self.store.rest, obj, obj.rest), sorting)
                sink.endAnonymous(auPair(sub), auPair(pre))
            return

        if isinstance(obj, N3Set):
            a = self.context.newBlankNode()
            tempobj = [mm for mm in obj] #I hate sorting things - yosi
            tempobj.sort(Term.compareAnyTerm)
            tempList = self.store.newList(tempobj)
            sink.startAnonymous(auxPairs((triple[CONTEXT],
                                        triple[PRED], triple[SUBJ], a)))
            self.dumpStatement(sink, (context, self.store.owlOneOf,
                                        a, tempList), sorting)
            sink.endAnonymous(auPair(sub), auPair(pre))
            return

        _anon, _incoming = self._topology(obj, context)

        if _anon and _incoming == 1:  # Embedded anonymous node in N3
            sink.startAnonymous(auxPairs(triple))
            ss = context.statementsMatching(subj=obj)
            if sorting: ss.sort(StoredStatement.comparePredObj)
            for t in ss:
                self.dumpStatement(sink, t.quad, sorting)
            sink.endAnonymous(sub.asPair(), pre.asPair())
            return

        self._outputStatement(sink, triple)


BNodePossibles = None   
def canItbeABNode(formula, symbol):
    def returnFunc():
        if BNodePossibles is not None:
            return symbol in BNodePossibles
        for quad in formula.statements:
            for s in PRED, SUBJ, OBJ:
                if isinstance(quad[s], Formula):
                    if BNodePossibles is None:
                        BNodePossible = quad[s].occurringIn(
                                                formula.existentials())
                    else:
                        BNodePossible.update(quad[s].occurringIn(
                                                formula.existentials()))
        return symbol in BNodePossibles
    return returnFunc


#ends

