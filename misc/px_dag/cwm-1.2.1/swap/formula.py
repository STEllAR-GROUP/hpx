from __future__ import generators
#! /usr/bin/python
"""

$Id: formula.py,v 1.63 2007/12/16 00:22:59 syosi Exp $

Formula
See:  http://www.w3.org/DesignIssues/Notation3

Interfaces
==========

The store stores many formulae, where one formula is what in
straight RDF implementations is known as a "triple store".
So look at the Formula class for a triple store interface.

See also for comparison, a python RDF API for the Redland library (in C):
   http://www.redland.opensource.ac.uk/docs/api/index.html 
and the redfoot/rdflib interface, a python RDF API:
   http://rdflib.net/latest/doc/triple_store.html

"""

__version__ = '$Id: formula.py,v 1.63 2007/12/16 00:22:59 syosi Exp $'[1:-1]

import types
import StringIO
import sys # for outputstrings. shouldn't be here - DWC

from set_importer import Set, ImmutableSet, sorted

import notation3    # N3 parsers and generators, and RDF generator

import diag  # problems importing the tracking flag, must be explicit it seems diag.tracking
from diag import progress, verbosity, tracking
from term import matchSet, \
    AnonymousNode , AnonymousExistential, AnonymousUniversal, \
    Term, CompoundTerm, List, \
    unifySequence, unify

from RDFSink import Logic_NS
from RDFSink import CONTEXT, PRED, SUBJ, OBJ
from RDFSink import FORMULA, SYMBOL



from why import Because, isTopLevel


###################################### Forumula
#
# An atomic forumla is three terms holds(s, p, o)
# A formula is
#  - an atomic formula
#  - a conjuction of formulas
#  - exists(x) F where F is a formula
#  - forall(x) F where F is a formula
# These are kept in a normal form:
#  exists e1,e2,e3... forall u1,u2u3, ... 
#    holds(s, p, o) and holds(s2,p2,o2) and ...

class Formula(AnonymousNode, CompoundTerm):
    """A formula of a set of RDF statements, triples.
    
    (The triples are actually instances of StoredStatement.)
    Other systems such as jena and redland use the term "Model" for Formula.
    For rdflib, this is known as a TripleStore.
    Cwm and N3 extend RDF to allow a literal formula as an item in a triple.
    
    A formula is either open or closed.  Initially, it is open. In this
    state is may be modified - for example, triples may be added to it.
    When it is closed, note that a different interned version of itself
    may be returned. From then on it is a constant.
    
    Only closed formulae may be mentioned in statements in other formuale.
    
    There is a reopen() method but it is not recommended, and if desperate should
    only be used immediately after a close(). 
    """
    def __init__(self, store, uri=None):
        AnonymousNode.__init__(self, store, uri)
        self.canonical = None # Set to self if this has been canonicalized
        self.statements = []
        self._existentialVariables = Set()
        self._universalVariables = Set()
        self.stayOpen = 0   # If set, works as a knowledegbase, never canonicalized.
        self._redirections = {}  # Used for equalities

        
    def __repr__(self):
        if self.statements == []:
            return "{}"
        if len(self.statements) == 1:
            st = self.statements[0]
            return "{"+`st[SUBJ]`+" "+`st[PRED]`+" "+`st[OBJ]`+"}"

        s = Term.__repr__(self)
        return "{%i}" % len(self.statements)
        
    def classOrder(self):
        return  11  # Put at the end of a listing because it makes it easier to read

    def compareTerm(self, other):
        "Assume is also a Formula - see function compareTerm below"
        for f in self, other:
            if f.canonical is not f:
                progress("@@@@@ Comparing formula NOT canonical", `f`)
        s = self.statements
        o = other.statements
        ls = len(s)
        lo = len(o)
        if ls > lo: return 1
        if ls < lo: return -1

        for se, oe, in  ((list(self.universals()), list(other.universals())),
                            (list(self.existentials()), list(other.existentials()))
                        ):
            lse = len(se)
            loe = len(oe)
            if lse > loe: return 1
            if lse < loe: return -1
            se.sort(Term.compareAnyTerm)
            oe.sort(Term.compareAnyTerm)
            for i in range(lse):
                diff = se[i].compareAnyTerm(oe[i])
                if diff != 0: return diff

#               @@@@ No need - canonical formulae are always sorted
        s.sort() # forumulae are all the same
        o.sort()
        for i in range(ls):
            diff = cmp(s[i],o[i])
            if diff != 0: return diff
        return 0
        import why
        raise RuntimeError("%s\n%s" % (dict(why.proofsOf), self.debugString()))
        raise RuntimeError("Identical formulae not interned! Length %i: %s\n\t%s\n vs\t%s" % (
                    ls, `s`, self.debugString(), other.debugString()))


    def existentials(self):
        """Return a list of existential variables with this formula as scope.
        
        Implementation:
        we may move to an internal storage rather than these pseudo-statements"""
        return self._existentialVariables


    def universals(self):
        """Return a set of variables universally quantified with this formula as scope.

        Implementation:
        We may move to an internal storage rather than these statements."""
        return self._universalVariables
    
    def variables(self):
        """Return a set of all variables quantified within this scope."""
        return self.existentials() | self.universals()
        
    def size(self):
        """Return the number statements.
        Obsolete: use len(F)."""
        return len(self.statements)

    def __len__(self):
        """ How many statements? """
        return len(self.statements)

    def __iter__(self):
        """The internal method which allows one to iterate over the statements
        as though a formula were a sequence.
        """
        for s in self.statements:
            yield s

    def newSymbol(self, uri):
        """Create or reuse the internal representation of the RDF node whose uri is given
        
        The symbol is created in the same store as the formula."""
        return self.store.newSymbol(uri)

    def newList(self, list):
        return self.store.nil.newList(list)

    def newLiteral(self, str, dt=None, lang=None):
        """Create or reuse the internal representation of the RDF literal whose string is given
        
        The literal is created in the same store as the formula."""
        return self.store.newLiteral(str, dt, lang)

    def newXMLLiteral(self, doc):
        """Create or reuse the internal representation of the RDF literal whose string is given
        
        The literal is created in the same store as the formula."""
        return self.store.newXMLLiteral(doc)

    def intern(self, value):
        return self.store.intern(value)
        
    def newBlankNode(self, uri=None, why=None):
        """Create a new unnamed node with this formula as context.
        
        The URI is typically omitted, and the system will make up an internal idnetifier.
        If given is used as the (arbitrary) internal identifier of the node."""
        x = AnonymousExistential(self, uri)
        self._existentialVariables.add(x)
        return x

    
    def declareUniversal(self, v, key=None):
        if key is not AnonymousUniversal:
            raise RuntimeError("""We have now disallowed the calling of declareUniversal.
For future reference, use newUniversal
""")
        if verbosity() > 90: progress("Declare universal:", v)
        if v not in self._universalVariables:
            self._universalVariables.add(v)
            if self.occurringIn(Set([self.newSymbol(v.uriref())])):
                raise ValueError("Internal error: declareUniversal: %s?" % v)
    def declareExistential(self, v):
        if verbosity() > 90: progress("Declare existential:", v)
        if v not in self._existentialVariables:  # Takes time
            self._existentialVariables.add(v)
##            if self.occurringIn(Set([v])) and not v.generated():
##                raise ValueError("Are you trying to confuse me, declaring %s as an existential?" % v)
#       else:
#           raise RuntimeError("Redeclared %s in %s -- trying to erase that" %(v, self)) 
        
    def newExistential(self, uri=None, why=None):
        """Create a named variable existentially qualified within this formula
        
        See also: existentials()  and newBlankNode()."""
        if uri == None:
            raise RuntimeError("Please use newBlankNode with no URI")
            return self.newBlankNode()  # Please ask for a bnode next time
        return self.store.newExistential(self, uri, why=why)
    
    def newUniversal(self, uri=None, why=None):
        """Create a named variable universally qualified within this formula
        
        See also: universals()"""
        x = AnonymousUniversal(self, uri)
##      self._universalVariables.add(x)
        return x

    def newFormula(self, uri=None):
        """Create a new open, empty, formula in the same store as this one.
        
        The URI is typically omitted, and the system will make up an internal idnetifier.
        If given is used as the (arbitrary) internal identifier of the formula."""
        return self.store.newFormula(uri)

    def statementsMatching(self, pred=None, subj=None, obj=None):
        """Return a READ-ONLY list of StoredStatement objects matching the parts given
        
        For example:
        for s in f.statementsMatching(pred=pantoneColor):
            print "We've got one which is ", `s[OBJ]`
            
        If none, returns []
        """
        for s in self.statements:
            if ((pred == None or pred is s.predicate()) and
                    (subj == None or subj is s.subject()) and
                    (obj == None or obj is s.object())):
                yield s

    def contains(self, pred=None, subj=None, obj=None):
        """Return boolean true iff formula contains statement(s) matching the parts given
        
        For example:
        if f.contains(pred=pantoneColor):
            print "We've got one statement about something being some color"
        """
        for s in self.statements:
            if ((pred == None or pred is s.predicate()) and
                    (subj == None or subj is s.subject()) and
                    (obj == None or obj is s.object())):
                return 1
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
        for s in self.statements:
            if ((pred == None or pred is s.predicate()) and
                    (subj == None or subj is s.subject()) and
                    (obj == None or obj is s.object())):
                break
        else: return None
        if obj == None: return s.object()
        if subj == None: return s.subject()
        if pred == None: return s.predicate()
        raise ValueError("You must give one wildcard in (%s, %s, %s)" %(subj, pred, obj))


    def the(self, subj=None, pred=None, obj=None):
        """Return None or the value filing the blank in the called parameters
        
        This is just like any() except it checks that there is only
        one answer in the store. It wise to use this when you expect only one.
        
        color = f.the(pred=pantoneColor, subj=myCar)
        redCar = f.the(pred=pantoneColor, obj=red)
        """
        return self.any(subj, pred, obj) # @@check >1

    def each(self, subj=None, pred=None, obj=None):
        """Return a list of values value filing the blank in the called parameters
        
        Examples:
        colors = f.each(pred=pantoneColor, subj=myCar)
        
        for redthing in f.each(pred=pantoneColor, obj=red): ...
        
        """
        for s in self.statements:
            if ((pred == None or pred is s.predicate()) and
                    (subj == None or subj is s.subject()) and
                    (obj == None or obj is s.object())):
                if pred == None: yield s.predicate()
                elif subj == None: yield s.subject()
                elif obj == None: yield s.object()
                else: raise ValueError(
                  "You must give one wildcard in (%s, %s, %s)" %(subj, pred, obj))

    def searchable(self, subj=None, pred=None, obj=None):
        """A pair of the difficulty of searching and a statement iterator of found statements
        
        The difficulty is a store-portable measure of how long the store
        thinks (in arbitrary units) it will take to search.
        This will only be used for choisng which part of the query to search first.
        If it is 0 there is no solution to the query, we know now.
        
        In this implementation, we use the length of the sequence to be searched."""
        difficulty = 1
        for p in subj, pred, obj:
            if p == None:
                difficulty += 1
        return difficulty, self.statementsMatching(subj, pred, obj) # use lazy eval here


    def substitution(self, bindings, why=None, cannon=False, keepOpen=False):
        "Return this or a version of me with subsitution made"
        assert isinstance(bindings, dict)
        store = self.store
        if self in bindings:
            return bindings[self]
        oc = self.occurringIn(bindings.keys())
        if oc == Set(): return self # phew!

        y = store.newFormula()
        if verbosity() > 90: progress("substitution: formula"+`self`+" becomes new "+`y`,
                                    " because of ", oc)
        y.loadFormulaWithSubstitution(self, bindings, why=why)
        if keepOpen:
            return y
        return y.canonicalize(cannon=cannon)

    def loadFormulaWithSubstitution(self, old, bindings={}, why=None, cannon=False):
        """Load information from another formula, subsituting as we go
        returns number of statements added (roughly)"""
        total = 0
        subWhy=Because('I said so #1', why)
        bindings2 = bindings.copy()
        bindings3 = {}
        for v in old.universals():
            if v not in bindings:
                bindings3[v] = self.newUniversal(bindings.get(v, v))
        for v in old.existentials():
            self.declareExistential(bindings.get(v, v))
        bindings2[old] = self
        realStatementList = []
        for s in old.statements[:] :   # Copy list!
            subj=s[SUBJ].substitution(
                                 bindings2, why=subWhy, cannon=cannon).substitution(
                                    bindings3, why=subWhy, cannon=cannon)
            ### Make sure we don't keep making copies of the same formula from running
            ##  the same rule again and again
            if isTopLevel(self) and isinstance(subj, Formula) and not subj.reallyCanonical:
                subj = subj.reopen()
                subj = subj.canonicalize(cannon=True)
            if isinstance(subj, Formula):
                subj = subj.canonical
            if subj is not s[SUBJ]:
                bindings2[s[SUBJ]] = subj
            pred=s[PRED].substitution(
                                 bindings2, why=subWhy, cannon=cannon).substitution(
                                    bindings3, why=subWhy, cannon=cannon)
            if pred is not s[PRED]:
                bindings2[s[PRED]] = pred
            obj=s[OBJ].substitution(
                                 bindings2, why=subWhy, cannon=cannon).substitution(
                                    bindings3, why=subWhy, cannon=cannon)
            if isTopLevel(self) and isinstance(obj, Formula) and not obj.reallyCanonical:
                obj = obj.reopen()
                obj = obj.canonicalize(cannon=True)
            if isinstance(obj, Formula):
                obj = obj.canonical
            if obj is not s[OBJ]:
                ### Question to self: What is this doing?
                bindings2[s[OBJ]] = obj

            try:
                total += self.add(subj=subj,
                                  pred=pred,
                                  obj=obj,
                                  why=why)
                realStatementList.append((subj, pred, obj))
            except AssertionError:
                print 'subj=%s' % subj.debugString()
                print 'oldSubj=%s' % (s[SUBJ].debugString(),)
                print 'subj.canonical=%s' % subj.canonical.debugString()
                raise
        if diag.chatty_flag > 80:
            def thing2string(x):
                if isinstance(x, (tuple, list)):
                    return '[' + ', '.join([thing2string(y) for y in x]) + ']'
                if isinstance(x, List):
                    return '(' + ' '.join([thing2string(y) for y in x]) + ')'
                else:
                    return str(x)
            progress('We added the following triples: %s' % (''.join(['\n\t%s' % thing2string(x) for x in realStatementList]),))

        return bindings3, total

    def subSet(self, statements, why=None):
        f = self.newFormula()
        for s in statements:
            c, p, s, o = s.quad
            f.add(s, p, o, why=why)
            assert c is self

        uu = f.occurringIn(self.universals())
        ee = f.occurringIn(self.existentials())
        bindings = {}
        
        f = self.newFormula()   ## do it right this time, with vars
        for v in uu:
#           progress("&&&&& New universal is %s\n\t in %s" % (v.uriref(), f))
            bindings[v] = f.newUniversal(v)
#           progress("&&&&& Universals are %s\n\t in %s" % (f.universals(), f))
        for v in ee:
            f.declareExistential(v)
        for s in statements:
            c, p, s, o = s.quad
            f.add(s.substitution(bindings, why=why), p.substitution(bindings, why=why), o.substitution(bindings, why=why), why=why)
        return f.close()  # probably slow - much slower than statement subclass of formula

                
    def substituteEquals(self, bindings, newBindings):
        """Return this or a version of me with subsitution made
        
        Subsitution of = for = does NOT happen inside a formula,
        as the formula is a form of quotation."""
        return self

    def occurringIn(self, vars):
        "Which variables in the list occur in this?"
        set = Set()
        for s in self.statements:
            for p in PRED, SUBJ, OBJ:
                y = s[p]
                if y is self:
                    pass
                else:
                    set = set | y.occurringIn(vars)
        return set

    def renameVars(self):
        if self._renameVarsMaps:
            if self in self._renameVarsMaps[-1]:
                return self._renameVarsMaps[-1][self]
        # progress('Running renameVars of self=%s' % self.debugString())
        m2 = {}
        for triple in self:
            for node in triple.spo():
                if isinstance(node, Formula):
                    if node not in m2:
                        m2[node] = node.renameVars()
        m = {}
        n = {}
        F1 = self.newFormula()
        F1.loadFormulaWithSubstitution(self, m2, why=Because("Vars in subexpressions must be renamed"))
        for v in sorted(list(F1.existentials()), Term.compareAnyTerm):
            m[v] = F1.newBlankNode()
        for v in sorted(list(F1.universals()), Term.compareAnyTerm):
            n[v] = F1.newUniversal(v)
        e = F1.existentials()
        u = F1.universals()
        for var in m:
            e.remove(var)
        for var in n:
            u.remove(var)
        
        m.update(n)
        #progress('F1 is %s' % F1.debugString())
        #progress('bindings are %s' % m)

        retVal = F1.substitution(m, why=Because("Vars must be renamed"), cannon=False, keepOpen=True)
        self._renameVarsMaps.append(0)
        retVal = retVal.canonicalize()
        self._renameVarsMaps.pop()
        if self._renameVarsMaps:
            self._renameVarsMaps[-1][self] = retVal
            self._renameVarsMaps[-1][retVal] = retVal

        
        assert retVal.canonical is retVal, retVal
        #progress('...got %s' % retVal.debugString())
        return retVal

##    def renameVars(self):
##        return self

    def resetRenames(reset = True):
        if reset:
            if diag.chatty_flag > 20:
                progress("Resetting all renamed vars maps ---------------------------------")
            Formula._renameVarsMaps.append({})
        else:
            Formula._renameVarsMaps.pop()
    resetRenames = staticmethod(resetRenames)
    _renameVarsMaps = []

##    def unify(self, other, vars=Set([]), existentials=Set([]),  bindings={}):
##      """See Term.unify()
##      """
##      if diag.chatty_flag > 99: progress("Unifying formula %s with %s" %
##          (`self`, `other`))
##      if diag.chatty_flag > 139: progress("Self is %s\n\nOther is %s" %
##          (self.debugString(), other.debugString()))
##      if not isinstance(other, Formula): return []
##      if self is other: return [({}, None)]
##      if (len(self) != len(other)
##          or len(self. _existentialVariables) != len(other._existentialVariables)
##          or len(self. _universalVariables) != len(other._universalVariables)
##          ): return []
##          
##      ex = existentials | self.existentials()  # @@ Add unis to make var names irrelevant?
##      return unifySequence(
##          [Set(self.statements), self.universals(), self.existentials()],
##          [Set(other.statements), other.universals(), other.existentials()],
##           vars | self.existentials() | self.universals(),
##           existentials , bindings)
    
    def unifySecondary(self, other, env1, env2, vars,
                       universals, existentials,
                       n1Source, n2Source):
        MyStatements = ImmutableSet(self.statements)
        OtherStatements = ImmutableSet(other.statements)
        for x in unify(MyStatements, OtherStatements, env1, env2, vars,
                       universals | self.universals() | other.universals(),
                       existentials | self.existentials() | other.existentials(),
                       n1Source, n2Source):
            yield x
                    
    def n3EntailedBy(pattern, kb, vars=Set([]), existentials=Set([]),  bindings={}):
        """See Term.unify() and term.matchSet()
        
        KB is a stronger statement han other.
        Bindings map variables in pattern onto kb.
        Self n3-entails other.
        Criteria:  Subset of self statements must match other statements.
          Self's exisetntials must be subset of other's
          Self's universals must be superset.
        """

        if diag.chatty_flag > 99: progress("n3EntailedBy:  %s entailed by %s ?" %
            (`pattern`, `kb`))
        if diag.chatty_flag > 139: progress("Pattern is %s\n\nKB is %s" %
            (pattern.debugString(), kb.debugString()))
        assert isinstance(kb, Formula), kb 
        if pattern is kb: return [({}, None)]
        nbs = matchSet(Set(pattern.statements), Set(kb.statements),
                        vars | pattern.existentials(),
                        # | pattern.universals(),
                        bindings)
        if diag.chatty_flag > 99: progress("n3EntailedBy: match result: ", `nbs`)
        if nbs == []: return []
        res = []
        for nb, rea in nbs:
            # We have matched the statements, now the lists of vars.
            ke = Set([ nb.get(e,e) for e in kb.existentials()])
            ke = pattern.occurringIn(ke) #Only ones mentioned count
            pe = Set([ nb.get(e,e) for e in pattern.existentials()])
            if diag.chatty_flag > 99: progress("\tpe=%s; ke=%s" %(pe,ke))
            if not ke.issubset(pe): return [] # KB must be stronger - less e's
            ku = Set([ nb.get(v,v) for v in kb.universals()])
            pu = Set([ nb.get(v,v) for v in pattern.universals()])
            if diag.chatty_flag > 99: progress("\tpu=%s; ku=%s" %(pu,ku))
            if not pu.issubset(ku): return [] # KB stronger -  more u's
            if diag.chatty_flag > 99: progress("n3EntailwsBy: success with ", `nb`)
            res.append((nb, None))    # That works
        return res
            



    def bind(self, prefix, uri):
        """Give a prefix and associated URI as a hint for output
        
        The store does not use prefixes internally, but keeping track
        of those usedd in the input data makes for more human-readable output.
        """
        return self.store.bind(prefix, uri)

    def add(self, subj, pred, obj, why=None):
        """Add a triple to the formula.
        
        The formula must be open.
        subj, pred and obj must be objects as for example generated 
        by Formula.newSymbol() and newLiteral(), or else literal
        values which can be interned.
        why     may be a reason for use when a proof will be required.
        """
        if self.canonical != None:
            raise RuntimeError("Attempt to add statement to canonical formula "+`self`)

        self.store.size += 1

        s = StoredStatement((self, pred, subj, obj))
        
        self.statements.append(s)
       
        return 1  # One statement has been added  @@ ignore closure extras from closure
                    # Obsolete this return value? @@@ 
    
    def removeStatement(self, s):
        """Removes a statement The formula must be open.
        
        This implementation is alas slow, as removal of items from tha hash is slow.
        """
        assert self.canonical == None, "Cannot remove statement from canonical "+`self`
        self.store.size = self.store.size-1
        self.statements.remove(s)
        return
    
    def close(self):
        """No more to add. Please return interned value.
        NOTE You must now use the interned one, not the original!"""
        return self.canonicalize()

    def canonicalize(F):
        """If this formula already exists, return the master version.
        If not, record this one and return it.
        Call this when the formula is in its final form, with all its statements.
        Make sure no one else has a copy of the pointer to the smushed one.
         
        LIMITATION: The basic Formula class does NOT canonicalize. So
        it won't spot idenical formulae. The IndexedFormula will.
        """
        store = F.store
        if F.canonical != None:
            if verbosity() > 70:
                progress("Canonicalize -- @@ already canonical:"+`F`)
            return F.canonical
        # @@@@@@@@ no canonicalization @@ warning
        F.canonical = F
        return F


    def n3String(self, base=None, flags=""):
        "Dump the formula to an absolute string in N3"
        buffer=StringIO.StringIO()
        _outSink = notation3.ToN3(buffer.write,
                                      quiet=1, base=base, flags=flags)
        self.store.dumpNested(self, _outSink)
        return buffer.getvalue().decode('utf_8')

    def ntString(self, base=None, flags="bravestpun"):
        "Dump the formula to an absolute string in N3"
        buffer=StringIO.StringIO()
        _outSink = notation3.ToN3(buffer.write,
                                      quiet=1, base=base, flags=flags)
        self.store.dumpBySubject(self, _outSink)
        return buffer.getvalue().decode('utf_8')


    def rdfString(self, base=None, flags=""):
        "Dump the formula to an absolute string in RDF/XML"
        buffer=StringIO.StringIO()
        import toXML
        _outURI = 'http://example.com/'
        _outSink = toXML.ToRDF(buffer, _outURI, base=base, flags=flags)
        self.store.dumpNested(self, _outSink)
        return buffer.getvalue()

    def outputStrings(self, channel=None, relation=None):
        """Fetch output strings from store, sort and output

        To output a string, associate (using the given relation) with a key
        such that the order of the keys is the order in which you want the corresponding
        strings output.

        @@ what is this doing here??
        """
        if channel == None:
            channel = sys.stdout
        if relation == None:
            relation = self.store.intern((SYMBOL, Logic_NS + "outputString"))
        list = self.statementsMatching(pred=relation)  # List of things of (subj, obj) pairs
        pairs = []
        for s in list:
            pairs.append((s[SUBJ], s[OBJ]))
        pairs.sort(comparePair)
        for key, str in pairs:
            channel.write(str.string.encode('utf-8'))

    def reopen(self):
        """Make a formula which was once closed oopen for input again.
        
        NOT Recommended.  Dangers: this formula will be, because of interning,
        the same objet as a formula used elsewhere which happens to have the same content.
        You mess with this one, you mess with that one.
        Much better to keep teh formula open until you don't needed it open any more.
        The trouble is, the parsers close it at the moment automatically. To be fixed."""
        return self.store.reopen(self)


#    def includes(f, g, _variables=[],  bindings=[]):
#       """Does this formula include the information in the other?
#       
#       bindings is for use within a query.
#       """
#       from swap.query import testIncludes  # Nor a dependency we want to make from here
#       return  testIncludes(f, g, _variables=_variables,  bindings=bindings)

    def generated(self):
        """Yes, any identifier you see for this is arbitrary."""
        return 1

#    def asPair(self):
#        """Return an old representation. Obsolete"""
#        return (FORMULA, self.uriref())

    def subjects(self, pred=None, obj=None):
        """Obsolete - use each(pred=..., obj=...)"""
        for s in self.statementsMatching(pred=pred, obj=obj)[:]:
            yield s[SUBJ]

    def predicates(self, subj=None, obj=None):
        """Obsolete - use each(subj=..., obj=...)"""
        for s in self.statementsMatching(subj=subj, obj=obj)[:]:
            yield s[PRED]

    def objects(self, pred=None, subj=None):
        """Obsolete - use each(subj=..., pred=...)"""
        for s in self.statementsMatching(pred=pred, subj=subj)[:]:
            yield s[OBJ]



    def doesNodeAppear(self, symbol):
        """Does that particular node appear anywhere in this formula

        This function is necessarily recursive, and is useful for the pretty printer
        It will also be useful for the flattener, when we write it.
        """
        for quad in self.statements:
            for s in PRED, SUBJ, OBJ:
                val = 0
                if isinstance(quad[s], CompoundTerm):
                    val = val or quad[s].doesNodeAppear(symbol)
                elif quad[s] == symbol:
                    val = 1
                else:
                    pass
                if val == 1:
                    return 1
        return 0

    def freeVariables(self):
        retVal = Set()
        for statement in self:
            for node in statement.spo():
                retVal.update(node.freeVariables())
        retVal.difference_update(self.existentials())
        retVal.difference_update(self.universals())
        if self.canonical:
            self.freeVariablesCompute = self.freeVariables
            self.freeVariables = lambda : retVal.copy()
        return retVal.copy()


#################################################################################


class StoredStatement:
    """A statememnt as an element of a formula
    """
    def __init__(self, q):
        self.quad = q

    def __getitem__(self, i):   # So that we can index the stored thing directly
        return self.quad[i]

    def __repr__(self):
        return "{"+`self[SUBJ]`+" "+`self[PRED]`+" "+`self[OBJ]`+"}"

#   The order of statements is only for canonical output
#   We cannot override __cmp__ or the object becomes unhashable,
# and can't be put into a dictionary.

    def __cmp__(self, other):
        """Just compare SUBJ, Pred and OBJ, others the same
        Avoid loops by spotting reference to containing formula"""
        if self is other: return 0
        if not isinstance(other, StoredStatement):
            return cmp(self.__class__, other.__class__)
        sc = self.quad[CONTEXT]
        oc = other.quad[CONTEXT]
        for p in [SUBJ, PRED, OBJ]: # Note NOT internal order
            s = self.quad[p]
            o = other.quad[p]
            if s is sc:
                if o is oc: continue
                else: return -1  # @this is smaller than other formulae
            else:           
                if o is oc: return 1
            if s is not o:
                return s.compareAnyTerm(o)
        return 0

    def __hash__(self):
        return id(self)

    def comparePredObj(self, other):
        """Just compare P and OBJ, others the same"""
        if self is other: return 0
        sc = self.quad[CONTEXT]
        oc = other.quad[CONTEXT]
        for p in [PRED, OBJ]: # Note NOT internal order
            s = self.quad[p]
            o = other.quad[p]
            if s is sc:
                if o is oc: continue
                else: return -1  # @this is smaller than other formulae
            else:           
                if o is oc: return 1
            if s is not o:
                return s.compareAnyTerm(o)
        return 0


    def context(self):
        """Return the context of the statement"""
        return self.quad[CONTEXT]
    
    def predicate(self):
        """Return the predicate of the statement"""
        return self.quad[PRED]
    
    def subject(self):
        """Return the subject of the statement"""
        return self.quad[SUBJ]
    
    def object(self):
        """Return the object of the statement"""
        return self.quad[OBJ]

    def spo(self):
        return (self.quad[SUBJ], self.quad[PRED], self.quad[OBJ])

    def __len__(self):
        return 1

    def statements(self):
        return [self]

    def occurringIn(self, vars):
        "Which variables in the list occur in this?"
        set = Set()
        if verbosity() > 98: progress("----occuringIn: ", `self`)
        for p in PRED, SUBJ, OBJ:
            y = self[p]
            if y is self:
                pass
            else:
                set = set | y.occurringIn(vars)
        return set

    def existentials(self):
        return self.occuringIn(self.quad[CONTEXT].existentials())

    def universals(self):
        return self.occuringIn(self.quad[CONTEXT].universals())

##    def unify(self, other, vars=Set([]), existentials=Set([]),  bindings={}):
##      """See Term.unify()
##      """
##      if diag.chatty_flag > 99: progress("Unifying statement %s with %s" %
##          (`self`, `other`))
##      if not isinstance(other, StoredStatement): raise TypeError
##      return unifySequence([self[PRED], self[SUBJ], self[OBJ]],
##          [other[PRED], other[SUBJ], other[OBJ]], 
##          vars, existentials, bindings)

    def unify(self, other, env1, env2, vars,
                       universals, existentials,
                       n1Source=32, n2Source=32):
        return unify(self, other, env1, env2, vars,
                       universals, existentials,
                       n1Source, n2Source)
    
    def unifySecondary(self, other, env1, env2, vars,
                       universals, existentials,
                       n1Source, n2Source):
        return unifySequence([self[PRED], self[SUBJ], self[OBJ]],
            [other[PRED], other[SUBJ], other[OBJ]], env1, env2, vars,
                             universals, existentials,
                             n1Source, n2Source)


    def asFormula(self, why=None):
        """The formula which contains only a statement like this.
        
        When we split the statement up, we lose information in any existentials which are
        shared with other statements. So we introduce a skolem constant to tie the
        statements together.  We don't have access to any enclosing formula 
        so we can't express its quantification.  This @@ not ideal.
        
        This extends the StoredStatement class with functionality we only need with "why" module."""
        
        store = self.quad[CONTEXT].store
        c, p, s, o = self.quad
        f = store.newFormula()   # @@@CAN WE DO THIS BY CLEVER SUBCLASSING? statement subclass of f?
        f.add(s, p, o, why=why)
        uu = f.freeVariables().intersection(c.universals())
        ee = f.occurringIn(c.existentials())
        bindings = {}
        
        f = store.newFormula()   ## do it right this time, with vars
        for v in uu:
#           progress("&&&&& New universal is %s\n\t in %s" % (v.uriref(), f))
            bindings[v] = f.newUniversal(v)
#           progress("&&&&& Universals are %s\n\t in %s" % (f.universals(), f))
        for v in ee:
            f.declareExistential(v)
        f.add(s.substitution(bindings, why=why), p.substitution(bindings, why=why), o.substitution(bindings, why=why), why=why)
        return f.close()  # probably slow - much slower than statement subclass of formula



#ends

