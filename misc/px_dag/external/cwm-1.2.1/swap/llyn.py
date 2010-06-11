#! /usr/bin/python
"""
$Id: llyn.py,v 1.185 2007/12/11 21:18:08 syosi Exp $


RDF Store and Query engine

Logic Lookup: Yet another Name

(also, in Wales, a lake - a storage area at the centre of the valley?)

This is an engine which knows a certian amount of stuff and can manipulate it.
It is a (forward chaining) query engine, not an (backward chaining) inference engine:
that is, it will apply all rules it can
but won't figure out which ones to apply to prove something.  It is not
optimized particularly.

Used by cwm - the closed world machine.
See:  http://www.w3.org/DesignIssues/Notation3

Interfaces
==========

This store stores many formulae, where one formula is what in
straight RDF implementations is known as a "triple store".
So look at the Formula class for a triple store interface.

See also for comparison, a python RDF API for the Redland library (in C):
   http://www.redland.opensource.ac.uk/docs/api/index.html 
and the redfoot/rdflib interface, a python RDF API:
   http://rdflib.net/latest/doc/triple_store.html

    
Copyright ()  2000-2004 World Wide Web Consortium, (Massachusetts Institute
of Technology, European Research Consortium for Informatics and Mathematics,
Keio University). All Rights Reserved. This work is distributed under the
W3C Software License [1] in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.
"""

# emacsbug="""emacs got confused by long string above"""

from __future__ import generators
# see http://www.amk.ca/python/2.2/index.html#SECTION000500000000000000000

from set_importer import Set, ImmutableSet

import types
import string

import re
import StringIO
import sys
import time, xml
from warnings import warn


import urllib # for log:content
import md5, binascii  # for building md5 URIs

import uripath
from uripath import canonical

from sax2rdf import XMLtoDOM
import xml.dom.minidom

from why import smushedFormula, Premise, newTopLevelFormula, isTopLevel

import notation3    # N3 parsers and generators, and RDF generator
from webAccess import webget

import diag  # problems importing the tracking flag,
             # and chatty_flag must be explicit it seems: use diag.tracking

from diag import progress, verbosity, tracking
from term import BuiltIn, LightBuiltIn, RDFBuiltIn, HeavyBuiltIn, Function, \
    MultipleFunction, ReverseFunction, MultipleReverseFunction, \
    Literal, XMLLiteral, Symbol, Fragment, FragmentNil, Term, LabelledNode, \
    CompoundTerm, List, EmptyList, NonEmptyList, AnonymousNode, N3Set, \
    UnknownType
from formula import Formula, StoredStatement
import reify

from weakref import WeakValueDictionary

from query import think, applyRules, testIncludes
import webAccess
from webAccess import DocumentAccessError
from local_decimal import Decimal

from RDFSink import Logic_NS, RDFSink, forSomeSym, forAllSym
from RDFSink import CONTEXT, PRED, SUBJ, OBJ, PARTS, ALL4
from RDFSink import N3_nil, N3_first, N3_rest, OWL_NS, N3_Empty, N3_List, \
                    N3_li, List_NS
from RDFSink import RDF_NS_URI

from RDFSink import FORMULA, LITERAL, LITERAL_DT, LITERAL_LANG, ANONYMOUS, SYMBOL

from pretty import Serializer

LITERAL_URI_prefix = "data:application/rdf+n3-literal;"
Delta_NS = "http://www.w3.org/2004/delta#"
cvsRevision = "$Revision: 1.185 $"


# Magic resources we know about

from RDFSink import RDF_type_URI, DAML_sameAs_URI

from why import Because, BecauseBuiltIn, BecauseOfRule, \
    BecauseOfExperience, becauseSubexpression, BecauseMerge ,report

STRING_NS_URI = "http://www.w3.org/2000/10/swap/string#"
META_NS_URI = "http://www.w3.org/2000/10/swap/meta#"
INTEGER_DATATYPE = "http://www.w3.org/2001/XMLSchema#integer"
FLOAT_DATATYPE = "http://www.w3.org/2001/XMLSchema#double"
DECIMAL_DATATYPE = "http://www.w3.org/2001/XMLSchema#decimal"
BOOL_DATATYPE = "http://www.w3.org/2001/XMLSchema#boolean"

#reason=Namespace("http://www.w3.org/2000/10/swap/reason#")

META_mergedWith = META_NS_URI + "mergedWith"
META_source = META_NS_URI + "source"
META_run = META_NS_URI + "run"

doMeta = 0  # wait until we have written the code! :-)
    

class DataObject:
    """The info about a term in the context of a specific formula
    It is created by being passed the formula and the term, and is
    then accessed like a python dictionary of sequences of values. Example:
    
    F = myWorkingFormula
    x = F.theObject(pred=rdfType obj=fooCar)
    for y in x[color][label]
    """
    def __init__(context, term):
        self.context = context
        self.term = term
        
    def __getItem__(pred):   #   Use . or [] ?
        values = context.objects(pred=pred, subj=self.term)
        for v in value:
            yield DataObject(self.context, v)


def arg_hash(arg):
    if isinstance(arg, dict):
        g = []
        for k, v in arg.items():
            g.append((arg_hash(k), arg_hash(v)))
        return hash(tuple(g))
    if isinstance(arg, (tuple, list)):
        g = []
        for k in arg:
            g.append(arg_hash(k))
        return hash(tuple(g))
    if isinstance(arg, Set):
        g = []
        for k in arg:
            g.append(arg_hash(k))
        return hash(ImmutableSet(g))
    return hash(arg)

def memoize(f):
    mymap = {}
    def k(*args, **keywords):
        n = arg_hash((args, keywords))
        if n not in mymap:
            mymap[n] = f(*args, **keywords)
        else:
            if diag.chatty_flag > 10:
                progress("momoizing helped!")
        return mymap[n]
    return k


####
###  BCHAIN @@@ Is this the right way to do it?
####
BACKWARD_CHAINING = False
class VARHASH(object):
    def __repr__(self): return 'VAR'
    def __str__(self): return 'VAR'
VARHASH = VARHASH()
####
###  /BCHAIN
####


###################################### Forumula
#
class IndexedFormula(Formula):
    """A formula which has indexes to facilitate queries.
    
    A formula is either open or closed.  Initially, it is open. In this
    state is may be modified - for example, triples may be added to it.
    When it is closed, note that a different interned version of itself
    may be returned. From then on it is a constant.
    
    Only closed formulae may be mentioned in statements in other formuale.
    
    There is a reopen() method but it is not recommended, and if desperate should
    only be used immediately after a close().
    """
    def __init__(self, store, uri=None):
        Formula.__init__(self, store, uri)
#       self._redirections = {}
        self.descendents = None   # Placeholder for list of closure under subcontext
#       self.collector = None # Object collecting evidence, if any 
        self._newRedirections = {}  # not subsituted yet
        self._index = {}
        self._index[(None,None,None)] = self.statements

        self._closureMode = ""
        self._closureAgenda = []
        self._closureAlready = []
        self.reallyCanonical = False


    def statementsMatching(self, pred=None, subj=None, obj=None):
        """Return a READ-ONLY list of StoredStatement objects matching the parts given
        
        For example:
        for s in f.statementsMatching(pred=pantoneColor):
            print "We've got one which is ", `s[OBJ]`
            
        If none, returns []
        """
        return self._index.get((pred, subj, obj), [])

    def contains(self, pred=None, subj=None, obj=None):
        """Return boolean true iff formula contains statement(s) matching the parts given
        
        For example:
        if f.contains(pred=pantoneColor):
            print "We've got one statement about something being some color"
        """
        x =  self._index.get((pred, subj, obj), [])
        if x : return 1
        return 0


    def any(self, subj=None, pred=None, obj=None):
        """Return None or the value filing the blank in the called parameters.
        
        Specifiy exactly two of the arguments.
        color = f.any(pred=pantoneColor, subj=myCar)
        somethingRed = f.any(pred=pantoneColor, obj=red)
        
        Note difference from the old store.any!!
        Note SPO order not PSO.
        To aboid confusion, use named parameters.
        """
        hits = self._index.get((pred, subj, obj), [])
        if not hits: return None
        s = hits[0]
        if pred is None: return s[PRED]
        if subj is None: return s[SUBJ]
        if obj is None: return s[OBJ]
        raise ParameterError("You must give one wildcard")


    def the(self, subj=None, pred=None, obj=None):
        """Return None or the value filing the blank in the called parameters
        
        This is just like any() except it checks that there is only
        one answer in the store. It wise to use this when you expect only one.
        
        color = f.the(pred=pantoneColor, subj=myCar)
        redCar = f.the(pred=pantoneColor, obj=red)
        """
        hits = self._index.get((pred, subj, obj), [])
        if not hits: return None
        assert len(hits) == 1, """There should only be one match for (%s %s %s).
            Found: %s""" %(subj, pred, obj, self.each(subj, pred, obj))
        s = hits[0]
        if pred is None: return s[PRED]
        if subj is None: return s[SUBJ]
        if obj is None: return s[OBJ]
        raise parameterError("You must give one wildcard using the()")

    def each(self, subj=None, pred=None, obj=None):
        """Return a list of values value filing the blank in the called parameters
        
        Examples:
        colors = f.each(pred=pantoneColor, subj=myCar)
        
        for redthing in f.each(pred=pantoneColor, obj=red): ...
        
        """
        hits = self._index.get((pred, subj, obj), [])
        if hits == []: return []
        if pred is None: wc = PRED
        elif subj is None: wc = SUBJ
        elif obj is None: wc = OBJ
        else: raise ParameterError("You must give one wildcard None for each()")
        res = []
        for s in hits:
            res.append(s[wc])   # should use yeild @@ when we are ready
        return res

    def searchable(self, subj=None, pred=None, obj=None):
        """A pair of the difficulty of searching and a statement iterator of found statements
        
        The difficulty is a store-portable measure of how long the store
        thinks (in arbitrary units) it will take to search.
        This will only be used for choisng which part of the query to search first.
        If it is 0 there is no solution to the query, we know now.
        
        In this implementation, we use the length of the sequence to be searched."""
        res = self._index.get((pred, subj, obj), [])
#        progress("searchable  %s, %s" %(self.statements, (pred, subj, obj))
        return len(res), res


    def add(self, subj, pred, obj, why=None):
        """Add a triple to the formula.
        
        The formula must be open.
        subj, pred and obj must be objects as for example generated by Formula.newSymbol()
        and newLiteral(), or else literal values which can be interned.
        why     may be a reason for use when a proof will be required.
        """
        if self.canonical != None:
            raise RuntimeError("Attempt to add statement to closed formula "+`self`)
        store = self.store
        
        if not isinstance(subj, Term): subj = store.intern(subj)
        if not isinstance(pred, Term): pred = store.intern(pred)
        if not isinstance(obj, Term): obj = store.intern(obj)
        newBindings = {}

#       Smushing of things which are equal into a single node
#       Even if we do not do this with owl:sameAs, we do with lists

        subj = subj.substituteEquals(self._redirections, newBindings)
        pred = pred.substituteEquals(self._redirections, newBindings)
        obj = obj.substituteEquals(self._redirections, newBindings)
            
        if diag.chatty_flag > 90:
            progress(u"Add statement (size before %i, %i statements) to %s:\n {%s %s %s}" % (
                self.store.size, len(self.statements),`self`,  `subj`, `pred`, `obj`) )
        if self.statementsMatching(pred, subj, obj):
            if diag.chatty_flag > 97:
                progress("Add duplicate SUPPRESSED %s: {%s %s %s}" % (
                    self,  subj, pred, obj) )
            return 0  # Return no change in size of store
            
        assert not isinstance(pred, Formula) or pred.canonical is pred, "pred Should be closed"+`pred`
        assert (not isinstance(subj, Formula)
                or subj is self
                or subj.canonical is subj), "subj Should be closed or self"+`subj`
        assert not isinstance(obj, Formula) or obj.canonical is obj, "obj Should be closed"+`obj`+`obj.canonical`
        store.size = store.size+1 # rather nominal but should be monotonic

        if False and isTopLevel(self):
            if isinstance(subj, Formula) and not subj.reallyCanonical:
                raise RuntimeError(subj.debugString())
            if isinstance(obj, Formula) and not obj.reallyCanonical:
                raise RuntimeError(obj.debugString())

# We collapse lists from the declared daml first,rest structure into List objects.
# To do this, we need a bnode with (a) a first; (b) a rest, and (c) the rest being a list.
# We trigger list collapse on any of these three becoming true.
# @@@ we don't reverse this on remove statement.  Remove statement is really not a user call.

# (Not clear: how t smush symbols without smushing variables. Need separate python class
# for variables I guess as everyone has been saying.
# When that happens, expend smushing to symbols.)

        if pred is store.rest:
            if isinstance(obj, List) and  subj in self._existentialVariables:
                ss = self.statementsMatching(pred=store.first, subj=subj)
                if ss:
                    s = ss[0]
                    self.removeStatement(s)
                    first = s[OBJ]
                    list = obj.prepend(first)
                    self._noteNewList(subj, list, newBindings)
                    self.substituteEqualsInPlace(newBindings, why=why)
                    return 1  # Added a statement but ... it is hidden in lists
    
        elif pred is store.first  and  subj in self._existentialVariables:
            ss = self.statementsMatching(pred=store.rest, subj=subj)
            if ss:
                s = ss[0]
                rest = s[OBJ]
                if isinstance(rest, List):
                    list = rest.prepend(obj)
                    self.removeStatement(s)
                    self._noteNewList(subj, list, newBindings)
                    self.substituteEqualsInPlace(newBindings)
                    return 1

        if pred is store.owlOneOf:
            if isinstance(obj, List) and subj in self._existentialVariables:
                new_set = store.newSet(obj)
                self._noteNewSet(subj, new_set, newBindings)
                self.substituteEqualsInPlace(newBindings)
                return 1

        if "e" in self._closureMode:
            if pred is store.sameAs:
                if subj is obj: return 0 # ignore a = a
                if obj in self.existentials() and subj not in self.existentials():
                    var, val = obj, subj
                elif ((subj in self.existentials() and obj not in self.existentials())
                    or (subj.generated() and not obj.generated())
                    or Term.compareAnyTerm(obj, subj) < 0): var, val = subj, obj
                else: var, val = obj, subj
                newBindings[var] = val
                if diag.chatty_flag > 90: progress("Equality: %s = %s" % (`var`, `val`))
                self.substituteEqualsInPlace(newBindings)               
                return 1

        if "T" in self._closureMode:
            if pred is store.type and obj is store.Truth:
                assert isinstance(subj, Formula), "What are we doing concluding %s is true?" % subj
                subj.resetRenames(True)
                self.loadFormulaWithSubstitution(subj.renameVars())
                subj.resetRenames(False)

#########
        if newBindings != {}:
            self.substituteEqualsInPlace(newBindings)
#######


        s = StoredStatement((self, pred, subj, obj))
        
        if diag.tracking:
            if (why is None): raise RuntimeError(
                "Tracking reasons but no reason given for"+`s`)
            report(s, why)

        # Build 8 indexes.
#       This now takes a lot of the time in a typical  cwm run! :-(
#       I honestly think the above line is a bit pessemistic. The below lines scale.
#       The above lines do not (removeStatement does not scale)

        if subj is self:  # Catch variable declarations
            if pred is self.store.forAll:
                if obj not in self._universalVariables:
                    if diag.chatty_flag > 50: progress("\tUniversal ", obj)
                    self._universalVariables.add(obj)
                return 1
            if pred is self.store.forSome:
                if obj not in self._existentialVariables:
                    if diag.chatty_flag > 50: progress("\tExistential ", obj)
                    self._existentialVariables.add(obj)
                return 1
            raise ValueError("You cannot use 'this' except as subject of forAll or forSome")

        self.statements.append(s)
       
        list = self._index.get((None, None, obj), None)
        if list is None: self._index[(None, None, obj)]=[s]
        else: list.append(s)

        list = self._index.get((None, subj, None), None)
        if list is None: self._index[(None, subj, None)]=[s]
        else: list.append(s)

        list = self._index.get((None, subj, obj), None)
        if list is None: self._index[(None, subj, obj)]=[s]
        else: list.append(s)

        list = self._index.get((pred, None, None), None)
        if list is None: self._index[(pred, None, None)]=[s]
        else: list.append(s)

        rein = self.newSymbol('http://dig.csail.mit.edu/2005/09/rein/network#requester')
        list = self._index.get((pred, None, obj), None)
        if list is None: self._index[(pred, None, obj)]=[s]
        else: list.append(s)

        list = self._index.get((pred, subj, None), None)
        if list is None: self._index[(pred, subj, None)]=[s]
        else: list.append(s)

        list = self._index.get((pred, subj, obj), None)
        if list is None: self._index[(pred, subj, obj)]=[s]
        else: list.append(s)

        if self._closureMode != "":
            self.checkClosure(subj, pred, obj)

        try:
            if self.isWorkingContext and diag.chatty_flag > 20:
                progress("adding",  (subj, pred, obj))
        except:
            pass

        return 1  # One statement has been added  @@ ignore closure extras from closure
                    # Obsolete this return value? @@@ 

                
    def removeStatement(self, s):
        """Removes a statement The formula must be open.
        
        This implementation is alas slow, as removal of items from tha hash is slow.
        The above statement is false. Removing items from a hash is easily over five times
        faster than removing them from a list.
        Also, truth mainainance is not done.  You can't undeclare things equal.
        This is really a low-level method, used within add() and for cleaning up the store
        to save space in purge() etc.
        """
        assert self.canonical is None, "Cannot remove statement from canonnical"+`self`
        self.store.size = self.store.size-1
        if diag.chatty_flag > 97:  progress("removing %s" % (s))
        context, pred, subj, obj = s.quad
        self.statements.remove(s)
        self._index[(None, None, obj)].remove(s)
        self._index[(None, subj, None)].remove(s)
        self._index[(None, subj, obj)].remove(s)
        self._index[(pred, None, None)].remove(s)
        self._index[(pred, None, obj)].remove(s)
        self._index[(pred, subj, None)].remove(s)
        self._index[(pred, subj, obj)].remove(s)
        #raise RuntimeError("The triple is %s: %s %s %s"%(context, pred, subj, obj))
        return

    def newCanonicalize(F):  ## Horrible name!
        if self._hashval is not None:
            return
        from term import Existential, Universal
        statements = []
        def convert(n):
            if isinstance(n, Universal):
                return Universal
            if isinstance(n, Existential):
                return Existential
            return n
        for statement in self.statements:
            statements.append(tuple([convert(x) for x in statement]))
        self._hashval = hash(ImmutableSet(statements))
    
    def canonicalize(F, cannon=False):
        """If this formula already exists, return the master version.
        If not, record this one and return it.
        Call this when the formula is in its final form, with all its
        statements.  Make sure no one else has a copy of the pointer to the
        smushed one.  In canonical form,
         - the statments are ordered
         - the lists are all internalized as lists
         
        Store dependency: Uses store._formulaeOfLength
        """
        if diag.chatty_flag > 70:
            progress('I got here')
        store = F.store
        if F.canonical != None:
            if diag.chatty_flag > 70:
                progress("End formula -- @@ already canonical:"+`F`)
            return F.canonical
        if F.stayOpen:
            if diag.chatty_flag > 70:
                progress("Canonicalizion ignored: @@ Knowledge base mode:"+`F`)
            return F

        F.store._equivalentFormulae.add(F)

        if (diag.tracking and isTopLevel(F)) or (F._renameVarsMaps and not cannon):  ## we are sitting in renameVars --- don't bother
            F.canonical = F
            return F
 
        fl = F.statements
        l = len(fl), len(F.universals()), len(F.existentials()) 
        possibles = store._formulaeOfLength.get(l, None)  # Any of same length

        if possibles is None:
            store._formulaeOfLength[l] = [F]
            if diag.chatty_flag > 70:
                progress("End formula - first of length", l, F)
            F.canonical = F
            F.reallyCanonical = True
 
            return F
        if diag.chatty_flag > 70:
            progress('I got here, possibles = ', possibles)
        fl.sort()
        fe = F.existentials()
        #fe.sort(Term.compareAnyTerm)
        fu = F.universals ()
        #fu.sort(Term.compareAnyTerm)

        for G in possibles:
        
#           progress("Just checking.\n",
#                   "\n", "_"*80, "\n", F.debugString(),
#                   "\n", "_"*80, "\n", G.debugString(),
#                   )
            gl = G.statements
            gkey = len(gl), len(G.universals()), len(G.existentials())
            if gkey != l: raise RuntimeError("@@Key of %s is %s instead of %s"
                %(G, `gkey`, `l`))

            gl.sort()
            for se, oe, in  ((fe, G.existentials()),
                             (fu, G.universals())):
                if se != oe:
                    break
            
            for i in range(l[0]):
                for p in PRED, SUBJ, OBJ:
                    if (fl[i][p] is not gl[i][p]):
#                       progress("""Mismatch on part %i on statement %i.
#           Becaue  %s  is not   %s"""
#                       % (p, i, `fl[i][p].uriref()`, `gl[i][p].uriref()`),
#                               )
#                       for  x in (fl[i][p], gl[i][p]):
#                           progress("Class %s, id=%i" %(x.__class__, id(x)))
#
#                       if Formula.unify(F,G):
#                           progress("""Unifies but did not match on part %i on statement %i.
#                           Because  %s  is not   %s"""
#                                       % (p, i, `fl[i][p].uriref()`, `gl[i][p].uriref()`),
#                                   "\n", "_"*80, "\n", F.debugString(),
#                                   "\n", "_"*80, "\n", G.debugString()
#                                   )
#                           raise RuntimeError("foundOne")

                        break # mismatch
                else: #match one statement
                    continue
                break
            else: #match
#               if not Formula.unify(F,G):
#                   raise RuntimeError("Look the same but don't unify")
                if tracking: smushedFormula(F,G)
                if diag.chatty_flag > 70: progress(
                    "** End Formula: Smushed new formula %s giving old %s" % (F, G))
                F.canonical = G
                del(F)  # Make sure it ain't used again
                return G



        possibles.append(F)
        F.canonical = F
        F.reallyCanonical = True
        if diag.chatty_flag > 70:
            progress("End formula, a fresh one:"+`F`)
##            for k in possibles:
##                print 'one choice is'
##                print k.n3String()
##                print '--------\n--------'
##            raise RuntimeError(F.n3String())
        return F


    def reopen(self):
        """Make a formula which was once closed oopen for input again.
        
        NOT Recommended.  Dangers: this formula will be, because of interning,
        the same objet as a formula used elsewhere which happens to have the same content.
        You mess with this one, you mess with that one.
        Much better to keep teh formula open until you don't needed it open any more.
        The trouble is, the parsers close it at the moment automatically. To be fixed."""
        return self.store.reopen(self)

    def setClosureMode(self, x):
        self._closureMode = x

    def checkClosure(self, subj, pred, obj):
        """Check the closure of the formula given new contents
        
        The s p o flags cause llyn to follow those parts of the new statement.
        i asks it to follow owl:imports
        r ask it to follow doc:rules
        """
        firstCall = (self._closureAgenda == [])
        if "s" in self._closureMode: self.checkClosureOfSymbol(subj)
        if "p" in self._closureMode: self.checkClosureOfSymbol(pred)
        if ("o" in self._closureMode or
            "t" in self._closureMode and pred is self.store.type):
            self.checkClosureOfSymbol(obj)
        if (("r" in self._closureMode and
              pred is self.store.docRules) or
            ("i" in self._closureMode and
              pred is self.store.imports)):   # check subject? @@@  semantics?
            self.checkClosureDocument(obj)
        if firstCall:
            while self._closureAgenda != []:
                x = self._closureAgenda.pop()
                self._closureAlready.append(x)
                x.dereference("m" + self._closureMode, self)
    
    def checkClosureOfSymbol(self, y):
        if not isinstance(y, Fragment): return
        return self.checkClosureDocument(y.resource)

    def checkClosureDocument(self, x):
        if x != None and x not in self._closureAlready and x not in self._closureAgenda:
            self._closureAgenda.append(x)


    def outputStrings(self, channel=None, relation=None):
        """Fetch output strings from store, sort and output

        To output a string, associate (using the given relation) with a key
        such that the order of the keys is the order in which you want the corresponding
        strings output.
        """
        if channel is None:
            channel = sys.stdout
        if relation is None:
            relation = self.store.intern((SYMBOL, Logic_NS + "outputString"))
        list = self.statementsMatching(pred=relation)  # List of things of (subj, obj) pairs
        pairs = []
        for s in list:
            pairs.append((s[SUBJ], s[OBJ]))
        pairs.sort(comparePair)
        for key, str in pairs:
            if not hasattr(str, "string"):
                print `str`
            channel.write(str.string.encode('utf-8'))


    def debugString(self, already=[]):
        """A simple dump of a formula in debug form.
        
        This formula is dumped, using ids for nested formula.
        Then, each nested formula mentioned is dumped."""
        red = ""
        if self._redirections != {}: red = " redirections:" + `self._redirections`
        str = `self`+ red + unicode(id(self)) + " is {"
        for vv, ss in ((self.universals().copy(), "@forAll"),(self.existentials().copy(), "@forSome")):
            if vv != Set():
                str = str + " " + ss + " " + `vv.pop()`
                for v in vv:
                    str = str + ", " + `v`
                str = str + "."
        todo = []
        for s in self.statements:
            subj, pred, obj = s.spo()
            str = str + "\n%28s  %20s %20s ." % (`subj`, `pred`, `obj`)
            for p in PRED, SUBJ, OBJ:
                if (isinstance(s[p], CompoundTerm)
                    and s[p] not in already and s[p] not in todo and s[p] is not self):
                    todo.append(s[p])
        str = str+ "}.\n"
        already = already + todo + [ self ]
        for f in todo:
            str = str + "        " + f.debugString(already)
        return str

    def _noteNewList(self,  bnode, list, newBindings):
        """Note that we have a new list.
        
        Check whether this new list (given as bnode) causes other things to become lists.
        Set up redirection so the list is used from now on instead of the bnode.        
        Internal function.

        This function is extraordinarily slow, .08 seconds per call on reify/reify3.n3"""
        if diag.chatty_flag > 80: progress("New list was %s, now %s = %s"%(`bnode`, `list`, `list.value()`))
        if isinstance(bnode, List): return  ##@@@@@ why is this necessary? weid.
        newBindings[bnode] = list
        if diag.chatty_flag > 80: progress("...New list newBindings %s"%(`newBindings`))
        self._existentialVariables.discard(bnode)
        possibles = self.statementsMatching(pred=self.store.rest, obj=bnode)  # What has this as rest?
        for s in possibles[:]:
            L2 = s[SUBJ]
            ff = self.statementsMatching(pred=self.store.first, subj=L2)
            if ff != []:
                first = ff[0][OBJ]
                self.removeStatement(s) 
                self.removeStatement(ff[0])
                list2 = list.prepend(first)
                self._noteNewList(L2, list2, newBindings)
        possibleSets = self.statementsMatching(pred=self.store.owlOneOf, obj=bnode)
        if possibleSets:
            new_set = self.store.newSet(list)
        for s in possibleSets[:]:
            s2 = s[SUBJ]
            if s2 in self._existentialVariables:
                self.removeStatement(s)
                self._noteNewSet(s2, new_set, newBindings)
        return

    def _noteNewSet(self, bnode, set, newBindings):
        newBindings[bnode] = set
        if diag.chatty_flag > 80: progress("...New set newBindings %s"%(`newBindings`))
        self._existentialVariables.discard(bnode)

    def substituteEqualsInPlace(self, redirections, why=None):
        """Slow ... does not use indexes"""
        bindings = redirections
        while bindings != {}:
            self._redirections.update(bindings)
            newBindings = {}
            for s in self.statements[:]:  # take a copy!
                changed = 0
                quad = [self, s[PRED], s[SUBJ], s[OBJ]]
                for p in PRED, SUBJ, OBJ:
                    x = s[p]
                    y = x.substituteEquals(bindings, newBindings)
                    if y is not x:
                        if diag.chatty_flag>90: progress("Substituted %s -> %s in place" %(x, y))
                        changed = 1
                        quad[p] = y
                if changed:
                    self.removeStatement(s)
                    self.add(subj=quad[SUBJ], pred=quad[PRED], obj=quad[OBJ], why=why)
            bindings = newBindings
            if diag.chatty_flag>70: progress("Substitions %s generated %s" %(bindings, newBindings))
        return

##    def unify(self, other, vars=Set([]), existentials=Set([]),  bindings={}):
    def unifySecondary(self, other, env1, env2, vars,
                       universals, existentials,
                       n1Source, n2Source):
        if self.canonical and other.canonical and self.store._equivalentFormulae.connected(self, other):
            yield (env1, env2)
        else:
            from query import n3Equivalent, testIncludes
            freeVars = self.freeVariables()   ## We can't use these
            retVal = n3Equivalent(self, other, env1, env2, vars,
                       universals, existentials,
                       n1Source, n2Source) # \
    ##               and n3Entails(self, other, vars=vars, existentials=existentials, bindings=bindings)
            for (env11, env12) in retVal:
                if env11 == env1 and env12 == env2:
                    self.store._equivalentFormulae.merge(self, other)
                yield (env11, env12)

##    unify = memoize(unify)



def comparePair(self, other):
    "Used only in outputString"
    for i in 0,1:
        x = self[i].compareAnyTerm(other[i])
        if x != 0:
            return x





###############################################################################################
#
#                       C W M - S P E C I A L   B U I L T - I N s
#
###########################################################################
    
# Equivalence relations

class BI_EqualTo(LightBuiltIn,Function, ReverseFunction):
    def eval(self,  subj, obj, queue, bindings, proof, query):
        return (subj is obj)   # Assumes interning

    def evalObj(self, subj, queue, bindings, proof, query):
        return subj

    def evalSubj(self, obj, queue, bindings, proof, query):
        return obj

class BI_notEqualTo(LightBuiltIn):
    def eval(self, subj, obj, queue, bindings, proof, query):
        return (subj is not obj)   # Assumes interning

BI_SameAs = BI_EqualTo

#### I hope this is never really added
##class BI_RunAsPython(LightBuiltIn, Function):
##    def evaluateObject(self, subject):
##        return eval(subject)

# Functions 
    
class BI_uri(LightBuiltIn, Function, ReverseFunction):

    def evalObj(self, subj, queue, bindings, proof, query):
        type, value = subj.asPair()
        if type == SYMBOL:
            return self.store.intern((LITERAL, value))

    def evaluateSubject(self, object):
        """Return the object which has this string as its URI
        
        #@@hm... check string for URI syntax?
        # or at least for non-uri chars, such as space?
        Note that relative URIs can be OK as the whole process
        has a base, which may be irrelevant. Eg see roadmap-test in retest.sh
        """
        store = self.store
        if ':' not in object:
            progress("Warning: taking log:uri of non-abs: %s" % object)
            return None
        #except (TypeError, AttributeError):
        #    return None
        return store.intern((SYMBOL, object))


class BI_dtlit(LightBuiltIn, Function):
    """built a datatype literal from a string and a uri"""
     
    def evaluateObject(self, subj_py):
        lex, dt = subj_py
        if dt is self.store.symbol("http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral"):
            try:
                dom = XMLtoDOM(lex)
            except SyntaxError, e:
                raise UnknownType # really malformed literal
            return self.store.newXMLLiteral(dom)
        else:
            return self.store.newLiteral(lex, dt)


class BI_rawUri(BI_uri):
    """This is like  uri except that it allows you to get the internal
    identifiers for anonymous nodes and formuale etc."""
     
    def evalObj(self, subj, queue, bindings, proof, query):
        type, value = subj.asPair()
        return self.store.intern((LITERAL, value))


class BI_rawType(LightBuiltIn, Function):
    """
    The raw type is a type from the point of view of the langauge: is
    it a formula, list, and so on. Needed for test for formula in finding subformulae
    eg see test/includes/check.n3 
    """

    def evalObj(self, subj,  queue, bindings, proof, query):
        store = self.store
        if isinstance(subj, Literal): y = store.Literal
        elif isinstance(subj, Formula): y = store.Formula
        elif isinstance(subj, List): y = store.List
        elif isinstance(subj, N3Set): y = store.Set
        elif isinstance(subj, AnonymousNode): y = store.Blank
        else: y = store.Other  #  None?  store.Other?
        if diag.chatty_flag > 91:
            progress("%s  rawType %s." %(`subj`, y))
        return y
        

class BI_racine(LightBuiltIn, Function):    # The resource whose URI is the same up to the "#" 

    def evalObj(self, subj,  queue, bindings, proof, query):
        if isinstance(subj, Fragment):
            return subj.resource
        else:
            return subj

# Heavy Built-ins

class BI_includes(HeavyBuiltIn):
    """Check that one formula does include the other.
    This limits the ability to bind a variable by searching inside another
    context. This is quite a limitation in some ways. @@ fix
    """
    def eval(self, subj, obj, queue, bindings, proof, query):
        store = subj.store
        if isinstance(subj, Formula) and isinstance(obj, Formula):
            return testIncludes(subj, obj, bindings=bindings) # No (relevant) variables
        return 0
            
    
class BI_notIncludes(HeavyBuiltIn):
    """Check that one formula does not include the other.

    notIncludes is a heavy function not only because it may take more time than
    a simple search, but also because it must be performed after other work so that
    the variables within the object formula have all been subsituted.  It makes no sense
    to ask a notIncludes question with variables, "Are there any ?x for which
    F does not include foo bar ?x" because of course there will always be an
    infinite number for any finite F.  So notIncludes can only be used to check, when a
    specific case has been found, that it does not exist in the formula.
    This means we have to know that the variables do not occur in obj.

    As for the subject, it does make sense for the opposite reason.  If F(x)
    includes G for all x, then G would have to be infinite.  
    """
    def eval(self, subj, obj, queue, bindings, proof, query):
        store = subj.store
        if isinstance(subj, Formula) and isinstance(obj, Formula):
            return not testIncludes(subj, obj,  bindings=bindings, interpretBuiltins=0) # No (relevant) variables
        return 0   # Can't say it *doesn't* include it if it ain't a formula

class BI_notIncludesWithBuiltins(HeavyBuiltIn):
    def eval(self, subj, obj, queue, bindings, proof, query):
        store = subj.store
        if isinstance(subj, Formula) and isinstance(obj, Formula):
            return not testIncludes(subj, obj,  bindings=bindings,
                    interpretBuiltins=1) # No (relevant) variables
        return 0   # Can't say it *doesn't* include it if it ain't a formula



class BI_semantics(HeavyBuiltIn, Function):
    """ The semantics of a resource are its machine-readable meaning, as an
    N3 forumula.  The URI is used to find a representation of the resource in
    bits which is then parsed according to its content type."""
    def evalObj(self, subj, queue, bindings, proof, query):
        store = subj.store
        if isinstance(subj, Fragment): doc = subj.resource
        else: doc = subj
        F = store.any((store._experience, store.semantics, doc, None))
        if F != None:
            if diag.chatty_flag > 10:
                progress("Already read and parsed "+`doc`+" to "+ `F`)
            return F

        if diag.chatty_flag > 10: progress("Reading and parsing " + doc.uriref())
        inputURI = doc.uriref()
#       if diag.tracking: flags="B"   # @@@@@@@@@@@ Yuk
#       else: flags=""
        F = self.store.load(inputURI, why=becauseSubexpression)
        if diag.chatty_flag>10: progress("    semantics: %s" % (F))
        return F.canonicalize()

class BI_semanticsWithImportsClosure(HeavyBuiltIn, Function):

    """ The semantics of a resource are its machine-readable meaning,
    as an N3 forumula.  The URI is used to find a representation of
    the resource in bits which is then parsed according to its content
    type.  Extension : It also loads all imported URIs"""
  
    def evalObj(self, subj, queue, bindings, proof, query):
        store = subj.store
        if isinstance(subj, Fragment): doc = subj.resource
        else: doc = subj
        F = store.any((store._experience, store.semanticsWithImportsClosure, doc, None))
        if F != None:
            if diag.chatty_flag > 10: progress("Already read and parsed "+`doc`+" to "+ `F`)
            return F
  
        if diag.chatty_flag > 10: progress("Reading and parsing with closure " + doc.uriref())
        inputURI = doc.uriref()
  
        F = store.newFormula()
        F.setClosureMode("i")
        F = store.load(uri=inputURI, openFormula=F)
          
        if diag.chatty_flag>10: progress("Reading and parsing with closure done.    semantics: %s" % (F))
#       if diag.tracking:
#            proof.append(F.collector)
        F = F.close()
        store.storeQuad((store._experience, store.semanticsWithImportsClosure, doc, F))
        return F
        
import httplib    
class BI_semanticsOrError(BI_semantics):
    """ Either get and parse to semantics or return an error message on any error """
    def evalObj(self, subj, queue, bindings, proof, query):
        import xml.sax._exceptions # hmm...
        store = subj.store
        x = store.any((store._experience, store.semanticsOrError, subj, None))
        if x != None:
            if diag.chatty_flag > 10: progress(`store._experience`+`store.semanticsOrError`+": Already found error for "+`subj`+" was: "+ `x`)
            return x
        try:
            return BI_semantics.evalObj(self, subj, queue, bindings, proof, query)
        except (IOError, SyntaxError, DocumentAccessError,
                xml.sax._exceptions.SAXParseException, httplib.BadStatusLine):
            message = sys.exc_info()[1].__str__()
            result = store.intern((LITERAL, message))
            if diag.chatty_flag > 0: progress(`store.semanticsOrError`+": Error trying to access <" + `subj` + ">: "+ message) 
            store.storeQuad((store._experience,
                             store.semanticsOrError,
                             subj,
                             result))
            return result
    

def loadToStore(term, types):
    """load content from the web and keep it in the store's experience.
    return resulting literal term
    
    raises IOError
    
    <DanC> the log:content built-in could keep an HTTP response
           object around a la tabulator too.
    <timbl> yes.
    <timbl> You learn a lot from a recode.
    """

    if isinstance(term, Fragment): doc = term.resource # i.e. racine
    else: doc = term
    store = term.store #hmm... separate store from term?
    C = store.any((store._experience, store.content, doc, None))
    if C != None:
        if diag.chatty_flag > 10: progress("already read " + `doc`)
        return C

    if diag.chatty_flag > 10: progress("Reading " + `doc`)
    inputURI = doc.uriref()

    netStream = webget(inputURI, types)

    #@@ get charset from headers
    str = netStream.read().decode('utf-8')
    C = store.intern((LITERAL, str))
    store.storeQuad((store._experience,
                     store.content,
                     doc,
                     C))
    return C

class BI_content(HeavyBuiltIn, Function):
    def evalObj(self, subj, queue, bindings, proof, query):
        try:
            return loadToStore(subj, [])
        except IOError:
            return None # hmm... is built-in API evolving to support exceptions?


class BI_xmlTree(HeavyBuiltIn, Function):
    def evalObj(self, subj, queue, bindings, proof, query):
        x= BI_content.evalObj(self, subj, queue, bindings, proof, query)
        dom = XMLtoDOM(x.value())
        return subj.store.intern((XMLLITERAL, dom))

class BI_parsedAsN3(HeavyBuiltIn, Function):
    def evalObj(self, subj, queue, bindings, proof, query):
        store = subj.store
        if isinstance(subj, Literal):
            F = store.any((store._experience, store.parsedAsN3, subj, None))
            if F != None: return F
            if diag.chatty_flag > 10: progress("parsing " + subj.string[:30] + "...")

            inputURI = subj.asHashURI() # iffy/bogus... rather asDataURI? yes! but make more efficient
            p = notation3.SinkParser(store)
            p.startDoc()
            p.feed(subj.string.encode('utf-8')) #@@ catch parse errors
            F = p.endDoc()
            F = F.close()
            store._experience.add(subj=subj, pred=store.parsedAsN3, obj=F)
            return F

class BI_conclusion(HeavyBuiltIn, Function):
    """ Deductive Closure

    Closure under Forward Inference, equivalent to cwm's --think function.
    This is a function, so the object is calculated from the subject.
    """
    def evalObj(self, subj, queue, bindings, proof, query):
        store = subj.store
        if isinstance(subj, Formula):
            assert subj.canonical != None
            F = self.store.any((store._experience, store.cufi, subj, None))  # Cached value?
            if F != None:
                if diag.chatty_flag > 10: progress("Bultin: " + `subj`+ " cached log:conclusion " + `F`)
                return F

            F = self.store.newFormula()
            newTopLevelFormula(F)
            if diag.tracking:
                reason = Premise("Assumption of builtin", (subj, self))
#               reason = BecauseMerge(F, subj)
#               F.collector = reason
#               proof.append(reason)
            else: reason = None
            if diag.chatty_flag > 10: progress("Bultin: " + `subj`+ " log:conclusion " + `F`)
            self.store.copyFormula(subj, F, why=reason) # leave open
            think(F)
            F = F.close()
            assert subj.canonical != None
            
            self.store.storeQuad((store._experience, store.cufi, subj, F),
                    why=BecauseOfExperience("conclusion"))  # Cache for later
            return F

class BI_supports(HeavyBuiltIn):
    """A more managable version of log:conclusion
The real version of this should appear in query.py
    """
    def eval(self, subj, obj, queue, bindings, proof, query):
        pass

class BI_filter(LightBuiltIn, Function):
    """Filtering of formulae

    """
    def evalObj(self, subj, queue, bindings, proof, query):
        store = subj.store
        if not isinstance(subj, List):
            raise ValueError('I need a list of two formulae')
        list = [x for x in subj] 
        if len(list) != 2:
            raise ValueError('I need a list of TWO formulae')
        if diag.chatty_flag > 30:
            progress("=== begin filter of:" + `list`)
        # list = [bindings.get(a,a) for a in list]
        base, filter = list
        F = self.store.newFormula()
        if diag.tracking:
            pass
        else: reason = None
        applyRules(base, filter, F)
        F = F.close()
        if diag.chatty_flag > 30:
            progress("=== end filter of:" + `list` + "we got: " + `F`)
        return F

class BI_vars(LightBuiltIn, Function):
    """Get only the variables from a formula

    """
    def evalObj(self, subj, queue, bindings, proof, query):
        F = self.store.newFormula()
        #F.existentials().update(subj.existentials())
        F.universals().update(subj.universals())
        return F.close()

class BI_universalVariableName(RDFBuiltIn): #, MultipleFunction):
    """Is the object the name of a universal variable in the subject?
    Runs even without interpretBuitins being set.  
    Used internally in query.py for testing for 
    Can be used as a test, or returns a sequence of values."""

    def eval(self, subj, obj, queue, bindings, proof, query):
        if not isinstance(subj, Formula): return None
        s = str(obj)
        if diag.chatty_flag > 180:
            progress(`subj.universals()`)
        return obj in subj.universals()
        for v in subj.universals():
            if v.uriref() == s: return 1
        return 0

    def evalObj(self,subj, queue, bindings, proof, query):
        if not isinstance(subj, Formula): return None
        return [subj.newLiteral(x.uriref()) for x in subj.universals()]

class BI_existentialVariableName(RDFBuiltIn): #, MultipleFunction):
    """Is the object the name of a existential variable in the subject?
    Can be used as a test, or returns a sequence of values.
    Currently gives BNode names too.  Maybe we make sep function for that?"""
    def eval(self, subj, obj, queue, bindings, proof, query):
        if not isinstance(subj, Formula): return None
        s = str(obj)
        if obj not in subj.existentials() and diag.chatty_flag > 25:
            progress('Failed, which is odd. Subj="%s", Obj="%s"' % (subj.debugString(), obj.debugString()))
        return obj in subj.existentials()
        for v in subj.existentials():
            if v.uriref() == s: return 1
        return 0

    def evalObj(self,subj, queue, bindings, proof, query):
        if not isinstance(subj, Formula): return None
        rea = None
        return [subj.newLiteral(x.uriref()) for x in subj.existentials()]


class BI_enforceUniqueBinding(RDFBuiltIn):
    """Is the mapping from the variable in the subject to the name in the object unique?

    """
    def eval(self, subj, obj, queue, bindings, proof, query):
        if not isinstance(subj, Formula): return None
        s = str(obj)
        if subj not in query.backwardMappings:
            query.backwardMappings[subj] = s
            return True
        return query.backwardMappings[subj] == s

class BI_conjunction(LightBuiltIn, Function):      # Light? well, I suppose so.
    """ The conjunction of a set of formulae is the set of statements which is
    just the union of the sets of statements
    modulo non-duplication of course."""
    def evalObj(self, subj, queue, bindings, proof, query):
        subj_py = subj.value()
        if diag.chatty_flag > 50:
            progress("Conjunction input:"+`subj_py`)
            for x in subj_py:
                progress("    conjunction input formula %s has %i statements" 
                                                % (x, x.size()))
        F = self.store.newFormula()
        if diag.tracking:
            reason = Because("I said so #4")
            #reason = BecauseMerge(F, subj_py)
        else: reason = None
        for x in subj_py:
            if not isinstance(x, Formula): return None # Can't
            if (x.canonical == None): # Not closed! !!
                F.canonical != None
                progress("Conjunction input NOT CLOSED:"+`x`) #@@@
            self.store.copyFormula(x, F, why=reason)   #  No, that is 
            if diag.chatty_flag > 74:
                progress("    Formula %s now has %i" % (`F`,len(F.statements)))
        return F.canonicalize()

class BI_n3String(LightBuiltIn, Function):      # Light? well, I suppose so.
    """ The n3 string for a formula is what you get when you
    express it in the N3 language without using any URIs.
    Note that there is no guarantee that two implementations will
    generate the same thing, but whatever they generate should
    parse back using parsedAsN3 to exaclty the same original formula.
    If we *did* have a canonical form it would be great for signature
    A canonical form is possisble but not simple."""
    def evalObj(self, subj, queue, bindings, proof, query):
        if diag.chatty_flag > 50:
            progress("Generating N3 string for:"+`subj`)
        if isinstance(subj, Formula):
            return self.store.intern((LITERAL, subj.n3String()))

class BI_reification(HeavyBuiltIn, Function, ReverseFunction):
    """



    """
    def evalSubj(self, obj, queue, bindings, proof, query):
        f = obj.store.newFormula()
        return reify.dereification(obj, f, obj.store)

    def evalObj(self, subj, queue, bindings, proof, query):
        f = subj.store.newFormula()
        q = subj.reification(f, {}, why=None)
        f=f.close()
        self.store.storeQuad((self.store._experience, self.store.type, f, 3), why=BecauseOfExperience("SomethingOrOther"))
        return q

import weakref

class Disjoint_set(object):
    class disjoint_set_node(object):
        def __init__(self, val):
            self.value = weakref.ref(val)
            self.parent = None
            self.rank = 0
        def link(self, other):
            if other.parent or self.parent:
                raise RuntimeError
            if self.rank > other.rank:
                other.parent = self
            else:
                self.parent = other
                if self.rank == other.rank:
                    other.rank += 1
        def find_set(self):
            if self.parent is None:
                return self
            self.parent = self.parent.find_set()
            return self.parent
        def disjoint(self, other):
            return self() is not other()
        def connected(self, other):
            return self() is other()
        __call__ = find_set

    def __init__(self):
        self.map = weakref.WeakKeyDictionary()

    def add(self, f):
        self.map[f] = self.disjoint_set_node(f)

    def connected(self, f, g):
        return self.map[f].connected(self.map[g])

    def merge(self, s1, s2):
        s1 = self.map[s1]()
        s2 = self.map[s2]()
        s1.link(s2)
    
################################################################################################

class RDFStore(RDFSink) :
    """ Absorbs RDF stream and saves in triple store
    """

    def clear(self):
        "Remove all formulas from the store     @@@ DOESN'T ACTUALLY DO IT/BROKEN"
        self.resources = WeakValueDictionary()    # Hash table of URIs for interning things
#        self.formulae = []     # List of all formulae        
        self._experience = None   #  A formula of all the things program run knows from direct experience
        self._formulaeOfLength = {} # A dictionary of all the constant formuale in the store, lookup by length key.
        self._formulaeOfLengthPerWorkingContext = {}
        self.size = 0
        self._equivalentFormulae = Disjoint_set()
        
    def __init__(self, genPrefix=None, metaURI=None, argv=None, crypto=0):
        RDFSink.__init__(self, genPrefix=genPrefix)
        self.clear()
        self.argv = argv     # List of command line arguments for N3 scripts

        run = uripath.join(uripath.base(), ".RUN/") + `time.time()`  # Reserrved URI @@

        if metaURI != None: meta = metaURI
        else: meta = run + "meta#formula"
        self.reset(meta)


        # Constants, as interned:
        
        self.forSome = self.symbol(forSomeSym)
        self.integer = self.symbol(INTEGER_DATATYPE)
        self.float  = self.symbol(FLOAT_DATATYPE)
        self.boolean = self.symbol(BOOL_DATATYPE)
        self.decimal = self.symbol(DECIMAL_DATATYPE)
        self.forAll  = self.symbol(forAllSym)
        self.implies = self.symbol(Logic_NS + "implies")
        self.insertion = self.symbol(Delta_NS + "insertion")
        self.deletion  = self.symbol(Delta_NS + "deletion")
        self.means = self.symbol(Logic_NS + "means")
        self.asserts = self.symbol(Logic_NS + "asserts")
        
# Register Light Builtins:

        log = self.symbol(Logic_NS[:-1])   # The resource without the hash

# Functions:        

        log.internFrag("racine", BI_racine)  # Strip fragment identifier from string
        log.internFrag("dtlit", BI_dtlit)

        self.rawType =  log.internFrag("rawType", BI_rawType) # syntactic type, oneOf:
        log.internFrag("rawUri", BI_rawUri)
        self.Literal =  log.internFrag("Literal", Fragment) # syntactic type possible value - a class
        self.List =     log.internFrag("List", Fragment) # syntactic type possible value - a class
        self.Set =     log.internFrag("Set", Fragment) # syntactic type possible value - a class
        self.Formula =  log.internFrag("Formula", Fragment) # syntactic type possible value - a class
        self.Blank   =  log.internFrag("Blank", Fragment)
        self.Other =    log.internFrag("Other", Fragment) # syntactic type possible value - a class
        self.filter  =  log.internFrag("filter", BI_filter) # equivilent of --filter
        self.vars    =  log.internFrag("vars", BI_vars) # variables of formula
        
        self.universalVariableName = log.internFrag(
                            "universalVariableName", BI_universalVariableName)
        self.existentialVariableName = log.internFrag(
                            "existentialVariableName", BI_existentialVariableName)
        self.enforceUniqueBinding = log.internFrag(
                            "enforceUniqueBinding", BI_enforceUniqueBinding)
        log.internFrag("conjunction", BI_conjunction)
        
# Bidirectional things:
        log.internFrag("uri", BI_uri)
        log.internFrag("equalTo", BI_EqualTo)
        log.internFrag("notEqualTo", BI_notEqualTo)
        log.internFrag("reification", BI_reification)

        owl = self.symbol(OWL_NS[:-1])
        self.sameAs = owl.internFrag("sameAs", BI_SameAs)

# Heavy relational operators:

        self.includes =         log.internFrag( "includes", BI_includes)
        self.supports =         log.internFrag( "supports", BI_supports)

#        log.internFrag("directlyIncludes", BI_directlyIncludes)
        self.notIncludes = log.internFrag("notIncludes", BI_notIncludes)
        self.smartNotIncludes = log.internFrag("notIncludesWithBuiltins", BI_notIncludesWithBuiltins)
#        log.internFrag("notDirectlyIncludes", BI_notDirectlyIncludes)

#Heavy functions:

#        log.internFrag("resolvesTo", BI_semantics) # obsolete
        self.semantics = log.internFrag("semantics", BI_semantics)
        self.cufi = log.internFrag("conclusion", BI_conclusion)
        self.semanticsOrError = log.internFrag("semanticsOrError", BI_semanticsOrError)
        self.semanticsWithImportsClosure = log.internFrag("semanticsWithImportsClosure", BI_semanticsWithImportsClosure)
        self.content = log.internFrag("content", BI_content)
        self.parsedAsN3 = log.internFrag("parsedAsN3",  BI_parsedAsN3)
        self.n3ExprFor = log.internFrag("n3ExprFor",  BI_parsedAsN3) ## Obsolete
        log.internFrag("n3String",  BI_n3String)

# Remote service flag in metadata:

        self.definitiveService = log.internFrag("definitiveService", Fragment)
        self.definitiveDocument = log.internFrag("definitiveDocument", Fragment)
        self.pointsAt = log.internFrag("pointsAt", Fragment)  # This was EricP's

# Constants:

        self.Truth = self.symbol(Logic_NS + "Truth")
        self.Falsehood = self.symbol(Logic_NS + "Falsehood")
        self.type = self.symbol(RDF_type_URI)
        self.Chaff = self.symbol(Logic_NS + "Chaff")
        self.docRules = self.symbol("http://www.w3.org/2000/10/swap/pim/doc#rules")
        self.imports = self.symbol("http://www.w3.org/2002/07/owl#imports")
        self.owlOneOf = self.symbol('http://www.w3.org/2002/07/owl#oneOf')

# List stuff - beware of namespace changes! :-(

        from cwm_list import BI_first, BI_rest
        rdf = self.symbol(List_NS[:-1])
        self.first = rdf.internFrag("first", BI_first)
        self.rest = rdf.internFrag("rest", BI_rest)
        self.nil = self.intern(N3_nil, FragmentNil)
        self.Empty = self.intern(N3_Empty)
        self.li = self.intern(N3_li)
        self.List = self.intern(N3_List)

        import cwm_string  # String builtins
        import cwm_os      # OS builtins
        import cwm_time    # time and date builtins
        import cwm_math    # Mathematics
        import cwm_trigo   # Trignometry
        import cwm_times    # time and date builtins
        import cwm_maths   # Mathematics, perl/string style
        import cwm_list    # List handling operations
        import cwm_set     # Set operations
        import cwm_sparql  # builtins for sparql
        import cwm_xml     # XML Document Object Model operations
        cwm_string.register(self)
        cwm_math.register(self)
        cwm_trigo.register(self)
        cwm_maths.register(self)
        cwm_os.register(self)
        cwm_time.register(self)
        cwm_times.register(self)
        cwm_list.register(self)
        cwm_set.register(self)
        cwm_sparql.register(self)
        cwm_xml.register(self)
        import cwm_crypto  # Cryptography
        if crypto:
            if cwm_crypto.USE_PKC == 0:
                raise RuntimeError("Try installing pycrypto, and make sure it is in you PYTHONPATH")
        else:
            cwm_crypto.USE_PKC = 0       
        cwm_crypto.register(self)  # would like to anyway to catch bug if used but not available

    def newLiteral(self, str, dt=None, lang=None):
        "Interned version: generate new literal object as stored in this store"
        key = (str, dt, lang)
        result = self.resources.get(key, None)
        if result != None: return result
#       if dt is not None: dt = self.newSymbol(dt)
        assert dt is None or isinstance(dt, LabelledNode)
        if dt is not None and not isinstance(dt, Fragment):
            progress("Warning: <%s> is not a fragment!" % dt)
        result = Literal(self, str, dt, lang)
        self.resources[key] = result
        return result
        
    def newXMLLiteral(self, dom):
        # We do NOT intern these so they will NOT have 'is' same as '=='
        return XMLLiteral(self, dom)
        
    def newFormula(self, uri=None):
        return IndexedFormula(self, uri)

    def newSymbol(self, uri):
        return self.intern(RDFSink.newSymbol(self, uri))

    def newSet(self, iterator=[], context=None):
        new_set = N3Set(iterator)
        Term.__init__(new_set, self)
        return new_set

    def newBlankNode(self, context, uri=None, why=None):
        """Create or reuse, in the default store, a new unnamed node within the given
        formula as context, and return it for future use"""
        return context.newBlankNode(uri=uri)

    def newExistential(self, context, uri=None, why=None):
        """Create or reuse, in the default store, a new named variable
        existentially qualified within the given
        formula as context, and return it for future use"""
        return self.intern(RDFSink.newExistential(self, context, uri, why=why))
    
    def newUniversal(self, context, uri=None, why=None):
        """Create or reuse, in the default store, a named variable
        universally qualified within the given
        formula as context, and return it for future use"""
        return self.intern(RDFSink.newUniversal(self, context, uri, why=why))



###################

    def reset(self, metaURI): # Set the metaURI
        self._experience = self.newFormula(metaURI + "_formula")
        assert isinstance(self._experience, Formula)

    def load(store, uri=None, openFormula=None, asIfFrom=None, contentType=None, remember=1,
                    flags="", referer=None, why=None, topLevel=False):
        """Get and parse document.  Guesses format if necessary.

        uri:      if None, load from standard input.
        remember: if 1, store as metadata the relationship between this URI and this formula.
        
        Returns:  top-level formula of the parsed document.
        Raises:   IOError, SyntaxError, DocumentError
        
        This was and could be an independent function, as it is fairly independent
        of the store. However, it is natural to call it as a method on the store.
        And a proliferation of APIs confuses.
        """
        baseURI = uripath.base()
        givenOpenFormula = openFormula
        if openFormula is None:
            openFormula = store.newFormula()
        if topLevel:
            newTopLevelFormula(openFormula)
        if uri != None and openFormula==None and remember:
            addr = uripath.join(baseURI, uri) # Make abs from relative
            source = store.newSymbol(addr)
            F = store._experience.the(source, store.semantics)
            if F != None:
                if diag.chatty_flag > 40: progress("Using cached semantics for",addr)
                return F 
            F = webAccess.load(store, uri, openFormula, asIfFrom, contentType, flags, referer, why)  
            store._experience.add(
                    store.intern((SYMBOL, addr)), store.semantics, F,
                    why=BecauseOfExperience("load document"))
            return F
            
        return webAccess.load(store, uri, openFormula, asIfFrom, contentType, flags, \
                              referer=referer, why=why)  

    



    def loadMany(self, uris, openFormula=None, referer=None):
        """Get, parse and merge serveral documents, given a list of URIs. 
        
        Guesses format if necessary.
        Returns top-level formula which is the parse result.
        Raises IOError, SyntaxError
        """
        assert type(uris) is type([])
        if openFormula is None: F = self.newFormula()
        else:  F = openFormula
        f = F.uriref()
        for u in uris:
            F.reopen()  # should not be necessary
            self.load(u, openFormula=F, remember=0, referer=referer)
        return F.close()

    def genId(self):
        """Generate a new identifier
        
        This uses the inherited class, but also checks that we haven't for some pathalogical reason
        ended up generating the same one as for example in another run of the same system. 
        """
        while 1:
            uriRefString = RDFSink.genId(self)
            hash = string.rfind(uriRefString, "#")
            if hash < 0 :     # This is a resource with no fragment
                return uriRefString # ?!
            resid = uriRefString[:hash]
            r = self.resources.get(resid, None)
            if r is None: return uriRefString
            fragid = uriRefString[hash+1:]
            f = r.fragments.get(fragid, None)
            if f is None: return uriRefString
            if diag.chatty_flag > 70:
                progress("llyn.genid Rejecting Id already used: "+uriRefString)
                
    def checkNewId(self, urirefString):
        """Raise an exception if the id is not in fact new.
        
        This is useful because it is usfeul
        to generate IDs with useful diagnostic ways but this lays them
        open to possibly clashing in pathalogical cases."""
        hash = string.rfind(urirefString, "#")
        if hash < 0 :     # This is a resource with no fragment
            result = self.resources.get(urirefString, None)
            if result is None: return
        else:
            r = self.resources.get(urirefString[:hash], None)
            if r is None: return
            f = r.fragments.get(urirefString[hash+1:], None)
            if f is None: return
        raise ValueError("Ooops! Attempt to create new identifier hits on one already used: %s"%(urirefString))
        return


    def internURI(self, str, why=None):
        warn("use symbol()", DeprecationWarning, stacklevel=3)
        return self.intern((SYMBOL,str), why)

    def symbol(self, str, why=None):
        """Intern a URI for a symvol, returning a symbol object"""
        return self.intern((SYMBOL,str), why)

    
    def _fromPython(self, x, queue=None):
        """Takem a python string, seq etc and represent as a llyn object"""
        if isinstance(x, tuple(types.StringTypes)):
            return self.newLiteral(x)
        elif type(x) is types.LongType or type(x) is types.IntType:
            return self.newLiteral(str(x), self.integer)
        elif isinstance(x, Decimal):
            return self.newLiteral(str(x), self.decimal)
        elif isinstance(x, bool):
            return self.newLiteral(x and 'true' or 'false', self.boolean)
        elif isinstance(x, xml.dom.minidom.Document):
            return self.newXMLLiteral(x)
        elif type(x) is types.FloatType:
            if `x`.lower() == "nan":  # We can get these form eg 2.math:asin
                return None
            return self.newLiteral(`x`, self.float)
        elif isinstance(x, Set) or isinstance(x, ImmutableSet):
            return self.newSet([self._fromPython(y) for y in x])
        elif isinstance(x, Term):
            return x
        elif hasattr(x,'__getitem__'): #type(x) == type([]):
            return self.nil.newList([self._fromPython(y) for y in x])
        return x

    def intern(self, what, dt=None, lang=None, why=None, ):
        """find-or-create a Fragment or a Symbol or Literal or list as appropriate

        returns URISyntaxError if, for example, the URIref has
        two #'s.
        
        This is the way they are actually made.
        """

        if isinstance(what, Term): return what # Already interned.  @@Could mask bugs
        if type(what) is not types.TupleType:
            if isinstance(what, tuple(types.StringTypes)):
                return self.newLiteral(what, dt, lang)
#           progress("llyn1450 @@@ interning non-string", `what`)
            if type(what) is types.LongType:
                return self.newLiteral(str(what),  self.integer)
            if type(what) is types.IntType:
                return self.newLiteral(`what`,  self.integer)
            if type(what) is types.FloatType:
                return self.newLiteral(repr(what),  self.float)
            if isinstance(what,Decimal):
                return self.newLiteral(str(what), self.decimal)
            if isinstance(what, bool):
                return self.newLiteral(what and 'true' or 'false', self.boolean)
            if type(what) is types.ListType: #types.SequenceType:
                return self.newList(what)
            raise RuntimeError("Eh?  can't intern "+`what`+" of type: "+`what.__class__`)

        typ, urirefString = what

        if typ == LITERAL:
            return self.newLiteral(urirefString, dt, lang)
        if typ == LITERAL_DT:
            return self.newLiteral(urirefString[0], self.intern(SYMBOL, urirefString[1]))
        if typ == LITERAL_LANG:
            return self.newLiteral(urirefString[0], None, urirefString[1])
        else:
            urirefString = canonical(urirefString)
            assert ':' in urirefString, "must be absolute: %s" % urirefString


            hash = string.rfind(urirefString, "#")
            if hash < 0 :     # This is a resource with no fragment
                assert typ == SYMBOL, "If URI <%s>has no hash, must be symbol" % urirefString
                result = self.resources.get(urirefString, None)
                if result != None: return result
                result = Symbol(urirefString, self)
                self.resources[urirefString] = result
            
            else :      # This has a fragment and a resource
                resid = urirefString[:hash]
                if string.find(resid, "#") >= 0:
                    raise URISyntaxError("Hash in document ID - can be from parsing XML as N3! -"+resid)
                r = self.symbol(resid)
                if typ == SYMBOL:
                    if urirefString == N3_nil[1]:  # Hack - easier if we have a different classs
                        result = r.internFrag(urirefString[hash+1:], FragmentNil)
                    else:
                        result = r.internFrag(urirefString[hash+1:], Fragment)
                elif typ == ANONYMOUS:
                    result = r.internFrag(urirefString[hash+1:], AnonymousNode)
                elif typ == FORMULA:
                    raise RuntimeError("obsolete")
                    result = r.internFrag(urirefString[hash+1:], IndexedFormula)
                else: raise RuntimeError, "did not expect other type:"+`typ`
        return result

    def newList(self, value, context=None):
        return self.nil.newList(value)

#    def deleteFormula(self,F):
#        if diag.chatty_flag > 30: progress("Deleting formula %s %ic" %
#                                            ( `F`, len(F.statements)))
#        for s in F.statements[:]:   # Take copy
#            self.removeStatement(s)


    def reopen(self, F):
        if F.canonical is None:
            if diag.chatty_flag > 50:
                progress("reopen formula -- @@ already open: "+`F`)
            return F # was open
        if diag.chatty_flag > 00:
            progress("warning - reopen formula:"+`F`)
        key = len(F.statements), len(F.universals()), len(F.existentials())
        try:
            self._formulaeOfLength[key].remove(F)  # Formulae of same length
        except (KeyError, ValueError):
            pass
        F.canonical = None
        return F


    def bind(self, prefix, uri):

        if prefix != "":   #  Ignore binding to empty prefix
            return RDFSink.bind(self, prefix, uri) # Otherwise, do as usual.
    
    def makeStatement(self, tuple, why=None):
        """Add a quad to the store, each part of the quad being in pair form."""
        q = ( self.intern(tuple[CONTEXT]),
              self.intern(tuple[PRED]),
              self.intern(tuple[SUBJ]),
              self.intern(tuple[OBJ]) )
        if q[PRED] is self.forSome and isinstance(q[OBJ], Formula):
            if diag.chatty_flag > 97:  progress("Makestatement suppressed")
            return  # This is implicit, and the same formula can be used un >1 place
        self.storeQuad(q, why)
                    
    def makeComment(self, str):
        pass        # Can't store comments


    def any(self, q):
        """Query the store for the first match.
        
        Quad contains one None as wildcard. Returns first value
        matching in that position.
        """
        list = q[CONTEXT].statementsMatching(q[PRED], q[SUBJ], q[OBJ])
        if list == []: return None
        for p in ALL4:
            if q[p] is None:
                return list[0].quad[p]


    def storeQuad(self, q, why=None):
        """ intern quads, in that dupliates are eliminated.

        subject, predicate and object are terms - or atomic values to be interned.
        Builds the indexes and does stuff for lists.
        Deprocated: use Formula.add()         
        """
        
        context, pred, subj, obj = q
        assert isinstance(context, Formula), "Should be a Formula: "+`context`
        return context.add(subj=subj, pred=pred, obj=obj, why=why)
        

    def startDoc(self):
        pass

    def endDoc(self, rootFormulaPair):
        return




##########################################################################
#
# Output methods:
#
    def dumpChronological(self, context, sink):
        "Fast as possible. Only dumps data. No formulae or universals."
        pp = Serializer(context, sink)
        pp. dumpChronological()
        del(pp)
        
    def dumpBySubject(self, context, sink, sorting=1):
        """ Dump by order of subject except forSome's first for n3=a mode"""
        pp = Serializer(context, sink, sorting=sorting)
        pp. dumpBySubject()
        del(pp)
        

    def dumpNested(self, context, sink, flags=""):
        """ Iterates over all URIs ever seen looking for statements
        """
        pp = Serializer(context, sink, flags=flags)
        pp. dumpNested()
        del(pp)



##################################  Manipulation methods:
#
#  Note when we move things, then the store may shrink as they may
# move on top of existing entries and we don't allow duplicates.
#
#   @@@@ Should automatically here rewrite any variable name clashes
#  for variable names which occur in the other but not as the saem sort of variable
# Must be done by caller.

    def copyFormula(self, old, new, why=None):
        new.loadFormulaWithSubstitution(old, why=why)
        return
##      bindings = {old: new}
##      for v in old.universals():
##          new.declareUniversal(bindings.get(v,v))
##      for v in old.existentials():
##          new.declareExistential(bindings.get(v,v))
##        for s in old.statements[:] :   # Copy list!
##            q = s.quad
##            for p in CONTEXT, PRED, SUBJ, OBJ:
##                x = q[p]
##                if x is old:
##                    q = q[:p] + (new,) + q[p+1:]
##            self.storeQuad(q, why)
                

    def purge(self, context, boringClass=None):
        """Clean up intermediate results

    Statements in the given context that a term is a Chaff cause
    any mentions of that term to be removed from the context.
    """
        if boringClass is None:
            boringClass = self.Chaff
        for subj in context.subjects(pred=self.type, obj=boringClass):
            self.purgeSymbol(context, subj)

    def purgeSymbol(self, context, subj):
        """Purge all triples in which a symbol occurs.
        """
        total = 0
        for t in context.statementsMatching(subj=subj)[:]:
                    context.removeStatement(t)    # SLOW
                    total = total + 1
        for t in context.statementsMatching(pred=subj)[:]:
                    context.removeStatement(t)    # SLOW
                    total = total + 1
        for t in context.statementsMatching(obj=subj)[:]:
                    context.removeStatement(t)    # SLOW
                    total = total + 1
        if diag.chatty_flag > 30:
            progress("Purged %i statements with %s" % (total,`subj`))
        return total


#    def removeStatement(self, s):
#        "Remove statement from store"
#       return s[CONTEXT].removeStatement(s)

    def purgeExceptData(self, context):
        """Remove anything which can't be expressed in plain RDF"""
        uu = context.universals()
        for s in context.statements[:]:
            for p in PRED, SUBJ, OBJ:
                x = s[p]
                if x in uu or isinstance(x, Formula):
                    context.removeStatement(s)
                    break
        context._universalVariables.clear()  # Cheat! @ use API



class URISyntaxError(ValueError):
    """A parameter is passed to a routine that requires a URI reference"""
    pass



def isString(x):
    # in 2.2, evidently we can test for isinstance(types.StringTypes)
    #    --- but on some releases, we need to say tuple(types.StringTypes)
    return type(x) is type('') or type(x) is type(u'')

#####################  Register this module

from myStore import setStoreClass
setStoreClass(RDFStore)

#ends

