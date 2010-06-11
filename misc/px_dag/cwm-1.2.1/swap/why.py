#! /usr/bin/python
"""
$Id: why.py,v 1.53 2007/06/27 17:58:39 syosi Exp $

A class for storing the reason why something is known.
The dontAsk constant reason is used as a reason for the explanations themselves-
 we could make it more complicated here for the recursively minded but i don't
 see the need at the moment.

Assumes wwe are using the process-global store -- uses Namespace() @@@
"""

flagHelp = """
n           use iNdirect formulas when they get too large 
g           always give the :gives clause, even when not needed
"""

import string
#import re
#import StringIO
import sys
import weakref
from set_importer import Set, sorted

# import notation3    # N3 parsers and generators, and RDF generator
# import sax2rdf      # RDF1.0 syntax parser to N3 RDF stream

import urllib # for hasContent
import uripath # DanC's tested and correct one
import md5, binascii  # for building md5 URIs

from uripath import refTo
from myStore  import Namespace
from term import Literal, CompoundTerm, AnonymousNode, List
# from formula import Formula

import diag
from diag import verbosity, progress

REIFY_NS = 'http://www.w3.org/2004/06/rei#'

#from RDFSink import CONTEXT, PRED, SUBJ, OBJ, PARTS, ALL4
#from RDFSink import FORMULA, LITERAL, ANONYMOUS, SYMBOL
from RDFSink import runNamespace


rdf=Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
log=Namespace("http://www.w3.org/2000/10/swap/log#")
reason=Namespace("http://www.w3.org/2000/10/swap/reason#")


global  dontAsk
global  proofsOf
proofsOf = weakref.WeakKeyDictionary() # Track collectors for formulae

# origin = {}   # track (statement, formula) to reason


# Reporting functions called from te rest of the system:

def smushedFormula(F, G):
    """The formula F has been replaced by G
    
    Because this module tracks formula in stores, if ever the address
    of a formula is changed, that is (currently) when it is 
    canonicalized, then the fact must be reported here.
    """
    progress("why: Formula %s has been replaced by %s" %(F,G))
    pF = proofsOf[F]
    pG = proofsOf[G]
    proofsOf[G] = pF + pG
    raise RuntimeError("@@@@ temp flag - we got to line 71 of why.py")
    del proofsOf[F]

def newTopLevelFormula(f):
    if f not in proofsOf:
        proofsOf[f] = [ KBReasonTracker(f) ]

def isTopLevel(f):
    return f in proofsOf


def report(statement, why):
    """Report a new statement to the reason tracking software
    
    This module stores the reasons.  The store should call this
    method to report new data. See the KBReasonTracker class

    Most formulas don't need collectors. Let's see if we can change that.
    """
    if why is dontAsk: return
    
    f = statement.context()
    collectors = proofsOf.get(f, [])

    if collectors == []:
        return None
        if diag.chatty_flag>50:
            progress("Adding %s. New collector for  %s" % (statement, why))
        collector = KBReasonTracker(f)
        proofsOf[f] = [ collector ]

    elif len(collectors) != 1:
        raise RuntimeError("""More than one tracker for formula %s.
            Means formula must already have been closed, so shouldn't be
            added to.""" % f)
            
    else: collector = collectors[0]
    
    return collector.newStatement(statement, why)
    
def explainFormula(f, flags=""):
    "Return the explanation formula for f"
    tr = proofsOf.get(f, None)
    if tr is None:
        raise ValueError(
            "No tracker. This may happen if the formula is validly empty. f=%s, proofsOf=%s" % (f, dict(proofsOf)))
    if not tr:
        raise ValueError(dict(proofsOf))
    try:
        return tr[0].explanation(flags=flags)
    except:
        print f.debugString()
        raise


# Internal utility

def _giveTerm(x, ko):
    """Describe a term in a proof
    
    This reifies symbols and bnodes.  Internal utility
    """
    #"
    from formula import Formula
    if isinstance(x, Formula):
        b = ko.newBlankNode(why=dontAsk)
        ko.add(subj=x, pred=ko.store.sameAs, obj=b,
                        why=dontAsk)
        return b
    elif isinstance(x, (Literal, CompoundTerm)):
        return x
    elif isinstance(x, AnonymousNode):
        b = ko.newBlankNode(why=dontAsk)
        ko.add(subj=b, pred=ko.newSymbol(REIFY_NS+"nodeId"), obj=x.uriref(),
                        why=dontAsk)
        return b
    else:
        return x.reification(ko, why=dontAsk)

def _subsetFormula(ss):
    """Return a subset formula containing the given statements
    
    The statements are all in the same context.""" #.
    s = ss.pop()  # @@ better way to pick one?
    f=s.context()
    ss.add(s)
    g = f.newFormula()
    for s in ss:
        try:
            g.add(s.subject(), s.predicate(), s.object(), why=dontAsk)
        except:
            print s
            print s.__class__
            print s.quad
            raise
#       progress("&&&&&&&&&&&&&&&&&&&&&&& ", g.n3String()) #@@@@@@@@@@
    g._existentialVariables = g.occurringIn(f._existentialVariables)
    g._universalVariables = g.occurringIn(f._universalVariables)
    g = g.close()
    return g
    
class Reason:
    """The Reason class holds a reason for having some information.
    Well, its subclasses actually do hold data.  This class should not be used
    itself to make instances.  Reasons may be given to any functions which put
    data into stores, is tracking or proof/explanation generation may be
    required"""
    def __init__(self):
        self.me = {}
        return


    def meIn(self, ko):
        "The representation of this object in the formula ko"
        assert self.me.get(ko, None) is None
        me = ko.newBlankNode(why= dontAsk) # @@ ko-specific, not reentrant
        self.me[ko] = me
        return me

    def explain(self, ko, flags):
        """Describe this reason to an RDF store
        Returns the value of this object as interned in the store.
        """
        raise RuntimeError("What, no explain method for this class?")
        

MAX_INLINE_SIZE = 10

def formulaStandIn(self, ko,f, flags):
    if 'n' not in flags:
        return f
    if len(f) <= MAX_INLINE_SIZE:
        return f
    try:
        m = self[(ko,f)]
        #progress('cache hit, we save $$!')
        return m
    except KeyError:
        from formula import Formula, StoredStatement
        if 'l' in flags:
            standIn = ko.newBlankNode(why= dontAsk)
        else:
            self[id(self)] += 1
            standIn = ko.newSymbol(runNamespace()+'_efm' + str(self[id(self)]))
        self[(ko,f)] = standIn

        ko.add(subj=f, pred=ko.store.sameAs, obj=standIn, why=dontAsk)
        return standIn

formulaStandIn = formulaStandIn.__get__({})
formulaStandIn.im_self[id(formulaStandIn.im_self)] = 0

        
class KBReasonTracker(Reason):
    """A reason tracker tracks the reasons for the statements in its formula.
    
    Beware that when a new formula is
    interned, the proofsOf dict must be informed that its identity has changed.
    The ForumulaReason is informed of each statement added to the knowlege
    base.
    
    A knowledge base (open formula) is made from the addition of forumlae,
    which result from, for example parsing a document or doing inference.
    Within such added formulae, there are variables, including bnodes, which
    have a cetain scope.  It is impossible to consider the process
    as being one of simply adding statements, as the cross-reference of
    the vaiables within the add formuls mst be preserved. 
    Variable renaming may occur as thr formula is added. 
    
    When we track these operations for generating a proof, a proof reason
    such as an BecauseOfRule or BecauseOfData corresponds to an added formula.
    The KBReasonTracker tracks which statements in a  formula came from which
    addion operations.
    """
    def __init__(self, formula=None):
        Reason.__init__(self)
        self._string = str
        self.formula = formula
        if formula != None:
            proofsOf[formula] = [self]
        self.reasonForStatement = {}

        return


    def newStatement(self, s, why):
        if verbosity() > 80 and why is not dontAsk:
            progress("Believing %s because of %s"%(s, why))
        import formula
        for x in s.quad[1:]:
            if isinstance(x, formula.Formula):
                if x.canonical is not x:
                    raise RuntimeError(x)
        assert why is not self
        self.reasonForStatement[s]=why
        if isinstance(why, (Premise, BecauseOfRule, BecauseOfData)):
            why.statements.add(s)


    def explanation(self, ko=None, flags=""):
        """Produce a justification for this formula into the output formula
        
        Creates an output formula if necessary.
        returns it.
        (This is different from reason.explain(ko) which returns the reason)"""

        if ko == None: ko = self.formula.store.newFormula()
        ko.bind("n3", "http://www.w3.org/2004/06/rei#")
        ko.bind("log", "http://www.w3.org/2000/10/swap/log#")
        ko.bind("pr", "http://www.w3.org/2000/10/swap/reason#")
        ko.bind("run", runNamespace())
        me=self.explain(ko, flags)
        ko.add(me, rdf.type, reason.Proof, why=dontAsk)
        return ko
        
    def explain(self, ko, flags):
        me = self.me.get(ko, None)
        if me != None: return me  #  Only do this once
        me = self.meIn(ko)

        g = self.formula
        e = g.existentials()
        if g.occurringIn(e) != e: raise RuntimeError(g.debugString())
        
        qed = ko.newBlankNode(why= dontAsk)
        standIn = formulaStandIn(ko,self.formula, flags)
        ko.add(subj=me, pred=reason.gives, obj=standIn, why=dontAsk)
        
    
        statementsForReason = {}  # reverse index: group by reason
        for s, rea in sorted(self.reasonForStatement.items()):
            x = statementsForReason.get(rea, None)
            if x is None: statementsForReason[rea] = [s]
            else: x.append(s)
        if diag.chatty_flag > 29:
            progress(
                "Collector %s (->%s), explaining %i reasons for total of %i statements:-" %
                (self, me, len(statementsForReason), len(self.reasonForStatement)))
            progress("reasonForStatement", self.reasonForStatement)
            progress("statementsForReason", statementsForReason)
        # @@ Should special case (no conjunction) if only one r
        if len(statementsForReason) != 1:
            ko.add(subj=me, pred=rdf.type, obj=reason.Conjunction, why=dontAsk) 
        
        for r, ss in statementsForReason.items():
            assert r is not self, ("Loop in reasons!", self, id(self), s)
            try:
               r1 = r.explain(ko, flags=flags)
            except:
               print ss
               raise
            if diag.chatty_flag > 29:
                progress(
                "\tExplaining reason %s (->%s) for %i statements" % (r, r1, len(ss)))
                for s in ss: progress("\t  Statement %s" % (s))
            if len(statementsForReason) == 1:  ## No need for conjunction
                ko.substituteEqualsInPlace({r1: me}, why=dontAsk)
            else:
                ko.add(me, reason.component, r1, why=dontAsk)
        return me

class BecauseMerge(KBReasonTracker):
    """Because this formula is a merging of others"""
    def __init__(self, f, set):
        raise RuntimeError(f, set)
        KBReasonTracker.__init__(self, f)
        self.fodder = Set()

    def newStatement(self, s, why):  # Why isn't a reason here, it is the source
        if verbosity() > 80:progress("Merge: Believing %s because of merge"%(s))
        self.fodder.add(why)
        self.reasonForStatement[s] = why
        
    def explain(self, ko, flags):
        me = self.me.get(ko, None)
        if me != None: return me  #  Only do this once
        me = self.meIn(ko)
        qed = ko.newBlankNode(why= dontAsk)
        ko.add(subj=me, pred=rdf.type, obj=reason.Conjunction, why=dontAsk) 
        if 'g' in flags:
            standIn = formulaStandIn(ko,self.formula, flags=flags)
            ko.add(subj=me, pred=reason.gives, obj=standIn, why=dontAsk)
        for x in self.fodder:
            ko.add(subj=me, pred=reason.mergeOf, obj=x.explain(ko, flags), why=dontAsk) 
        return me

class BecauseSubexpression(Reason):
    """This was generated as part of a calculatio of a subexpression.
    
     It is is not necessarily believed"""

    def explain(self, ko, flags):
        """Describe this reason to an RDF store
        Returns the value of this reason as interned in the store.
        """
        me = self.me.get(ko, None)
        if me != None: return me  #  Only do this once
        me = self.meIn(ko)
        ko.add(subj=me, pred=rdf.type, obj=reason.TextExplanation, why=dontAsk)
        ko.add(subj=me, pred=reason.text, obj=ko.newLiteral("(Subexpression)"),
                    why=dontAsk)
        return me

becauseSubexpression = BecauseSubexpression()



class Because(Reason):
    """For the reason given on the string.
    This is a kinda end of the road reason.
    
    A nested reason can also be given.
    """
    def __init__(self, str, because=None):
        Reason.__init__(self)
        self._string = str
        self._reason = because
        return

    def explain(self, ko, flags):
        """Describe this reason to an RDF store
        Returns the value of this reason as interned in the store.
        """
        raise RuntimeError("An internal Object slipped out, but we caught it.")
        me = self.me.get(ko, None)
        if me != None: return me  #  Only do this once
        me = self.meIn(ko)
        ko.add(subj=me, pred=rdf.type, obj=reason.TextExplanation, why=dontAsk)
        ko.add(subj=me, pred=reason.text, obj=ko.newLiteral(self._string),
                                why=dontAsk)
        if self._reason is not None:
            ko.add(subj=me, pred=reason.because, obj=self._reason.explain(ko, flags=flags),
                                                        why=dontAsk)
        return me
dontAsk = Because("Generating explanation")

class Premise(Reason):
    """For the reason given on the string.
    This is a kinda end of the road reason.
    It contais the info which was literally supplied as a premise.
    
    A nested reason can also be given.
    Because a premise has to be taken for granted, the tracker
    has to tell a Premis what statements it has.
    """
    def __init__(self, str, because=None):
        Reason.__init__(self)
        self._string = str
        self._reason = because
        self.statements = Set()
        return

    def explain(self, ko, flags):
        """Describe this reason to an RDF store
        Returns the value of this reason as interned in the store.
        """
        me = self.me.get(ko, None)
        if me != None: return me  #  Only do this once
        me = self.meIn(ko)
        if diag.chatty_flag>49: progress("Premise reason=%s ko=%s"%(self,me))
        ko.add(subj=me, pred=rdf.type, obj=reason.Premise, why=dontAsk)
        ko.add(subj=me, pred=reason.text, obj=ko.newLiteral(self._string),
                                why=dontAsk)

        if not self.statements:
            pass
##          raise RuntimeError("No given data for Premise %s" % self)
        else:
            prem = _subsetFormula(self.statements)
            standIn = formulaStandIn(ko,prem, flags=flags)
            ko.add(me, reason.gives, standIn, why=dontAsk)
            if diag.chatty_flag >59:
                progress("Premise (%s) is:\n%s" % 
                        ( self._string, prem.n3String()))
        return me



class BecauseOfRule(Reason):
    def __init__(self, rule, bindings, knownExistentials, evidence, kb, because=None):
        #print rule
        #raise Error
        Reason.__init__(self)
        self._bindings = bindings
        self._rule = rule
        self._evidence = evidence # Set of statements etc to justify LHS
        self._kb = kb # The formula the rule was trusting at base
        self._reason = because
        self._existentials = knownExistentials
        self.statements = Set()
        return


    def explain(self, ko, flags):
        """Describe this reason to an RDF store
        Returns the value of this reason as interned in the store.
        """
        me = self.me.get(ko, None)
        if me != None: return me  #  Only do this once
        me = self.meIn(ko)
        if diag.chatty_flag>49: progress("Inference=%s ko=%s"%(self,me))
        ko.add(subj=me, pred=rdf.type, obj=reason.Inference, why=dontAsk) 
        for var, (val, source) in self._bindings.items():
            b = ko.newBlankNode(why= dontAsk)
            ko.add(subj=me, pred=reason.binding, obj=b, why= dontAsk)
            ko.add(subj=b, pred=reason.variable,
                        obj=_giveTerm(var,ko),why= dontAsk)
            valObject = _giveTerm(val, ko)
            ko.add(subj=b, pred=reason.boundTo,
                        obj=valObject, why= dontAsk)
            if val in self._existentials:
                ko.add(subj=valObject, pred=rdf.type,
                       obj=reason.Existential, why=dontAsk)
            
        if diag.chatty_flag>49: progress("rule:")
        ru = explainStatement(self._rule,ko, flags=flags)
        ko.add(subj=me, pred=reason.rule, obj=ru, why=dontAsk)
            
        if diag.chatty_flag>49: progress("evidence:")
        ev = []  # For PML compatability we will store it as a collection
        realStatements = []
        for s in self._evidence:
            if isinstance(s, BecauseBuiltIn):
                try:
                    e = s.explain(ko, flags=flags)
                except:
                    print s
                    raise
                ev.append(e)
            else:
                f = s.context()
                if f is self._kb: # Normal case
                    ## We need something a little more complicated
                    realStatements.append(s)
                    if s.predicate() is f.store.includes:
                        for t in self.evidence:
                            if t.context() is s.subject():
                                progress("Included statement used:" + `t`)
                                try:
                                    ko.add(e, reason.includeEvidence,
                                    explainStatement(t, ko)) 
                                except:
                                    print s
                                    raise 
#               else:
#                   progress("Included statement found:" + `s`)
        ev.extend(explainStatements(realStatements, ko))
        ko.add(subj=me, pred=reason.evidence, obj=ev, why= dontAsk)
        if "g" in flags:
            prem = _subsetFormula(self.statements)
            standIn = formulaStandIn(ko,prem, flags=flags)
            ko.add(me, reason.gives, standIn, why=dontAsk)
            if diag.chatty_flag >59:
                progress("Rule (%s) is:\n%s" % 
                        ( self._string, prem.n3String()))
        return me



def getStatementReason(s):
    f = s.context()
    KBReasonTrackers = proofsOf.get(f, None)

    if KBReasonTrackers is None:
        raise RuntimeError(
        """Ooops, no reason collector for this formula?! 
        No proof for formula: %s
        Needed for statement: %s
        Only have proofs for %s.
        Formula contents as follows:
        %s
        """ % ( f, s, dict(proofsOf), f.debugString())) 

    tracker = KBReasonTrackers[0]

    return tracker.reasonForStatement.get(s, None)
    
def subFormulaStandIn(self, ko,f, flags):
    if 'n' not in flags:
        return f
    from formula import Formula
    if isinstance(f, List):
        return ko.store.newList([subFormulaStandIn(ko,x, flags) for x in f])
    if not isinstance(f, Formula) or len(f) <= MAX_INLINE_SIZE:
        return f
    try:
        m = self[(ko,f)]
        #progress('cache hit, we save $$!')
        return m
    except KeyError:
        if 'l' in flags:
            standIn = ko.newBlankNode(why= dontAsk)
        else:
            self[id(self)] += 1
            standIn = ko.newSymbol(runNamespace()+'_fm' + str(self[id(self)]))
        self[(ko,f)] = standIn
        ko.add(subj=f, pred=reason.representedBy, obj=standIn, why=dontAsk)
        return standIn

subFormulaStandIn = subFormulaStandIn.__get__({})
subFormulaStandIn.im_self[id(subFormulaStandIn.im_self)] = 0

def explainStatement(s, ko, ss=None, flags=""):
    si = describeStatement([s], ko, flags=flags)


    statementReason = getStatementReason(s)

    if statementReason == None:
        raise RuntimeError(
        """Ooops, no reason for this statement?! 
        Collector: %s
        Formula: %s
        No reason for statement: %s
        Reasons for statements we do have: %s
        Formula contents as follows:
        %s
        """ % (tracker, f, s, tracker.reasonForStatement,
            f.debugString()))   

    if diag.chatty_flag >49: progress("explaining statement: %s" % (s))
    ri = statementReason.explain(ko, flags=flags)
    ko.add(subj=si, pred=reason.because, obj=ri, why=dontAsk)
    return si

def explainStatements(s_l, ko, ss=None, flags=""):
    if not s_l:
        return []
    reasonsForStatement = {}
    f = s_l[0].context()
    for s in s_l:
        statementReason = getStatementReason(s)

        if statementReason == None:
            raise RuntimeError(
            """Ooops, no reason for this statement?! 
            Collector: %s
            Formula: %s
            No reason for statement: %s
            Reasons for statements we do have: %s
            Formula contents as follows:
            %s
            """ % (tracker, f, s, tracker.reasonForStatement,
                f.debugString()))
        reasonsForStatement.setdefault(statementReason, []).append(s)
    si_l = []
    for r, statements in reasonsForStatement.items():
        si = describeStatement(statements, ko, flags=flags)
        si_l.append(si)
        ri = r.explain(ko, flags=flags)
        ko.add(subj=si, pred=reason.because, obj=ri, why=dontAsk)
    return si_l
    

def describeStatement(s_l, ko, flags):
        "Describe the statement into the output formula ko"
        from formula import StoredStatement
        f = s_l[0].context()
        s_l2 = []
        for s in s_l:
            con, pred, subj, obj = s
            subj = subFormulaStandIn(ko, subj, flags)
            obj = subFormulaStandIn(ko, obj, flags)
            s = StoredStatement([con, pred, subj, obj])
            s_l2.append(s)
        
        si = ko.newBlankNode(why=dontAsk)
        ko.add(si, rdf.type, reason.Extraction, why=dontAsk)
        standIn = formulaStandIn(ko, f.subSet(s_l2, why=dontAsk), flags=flags)
        ko.add(si, reason.gives, standIn, why=dontAsk)
        return si

        

        
class BecauseOfData(Because):
    """Directly from data in the resource whose URI is the string.
    
    A nested reason can also be given, for why this resource was parsed.
    """
    pass

    def __init__(self, source, because=None):
        Reason.__init__(self)
        self._source = source
        self._reason = because
        self.statements = Set()
        return

    def explain(self, ko, flags):
        """Describe this reason to an RDF store
        Returns the value of this reason as interned in the store.
        """
        me = self.me.get(ko, None)
        if me != None: return me  #  Only do this once
        me = self.meIn(ko)
        if diag.chatty_flag>49: progress("Parsing reason=%s ko=%s"%(self,me))
        ko.add(subj=me, pred=rdf.type, obj=reason.Parsing, why=dontAsk)
        ko.add(subj=me, pred=reason.source, obj=self._source, why=dontAsk)
        if self._reason:
            ko.add(subj=me, pred=reason.because, 
                   obj=self._reason.explain(ko, flags=flags),
                   why=dontAsk)
        if "g" in flags:
            prem = _subsetFormula(self.statements)
            standIn = formulaStandIn(ko,prem, flags=flags)
            ko.add(me, reason.gives, standIn, why=dontAsk)
            if diag.chatty_flag >59:
                progress("Rule (%s) is:\n%s" % 
                        ( self._string, prem.n3String()))
        return me


class BecauseOfCommandLine(Because):
    """Because of the command line given in the string"""


    def explain(self, ko, flags):
        """Describe this reason to an RDF store
        Returns the value of this reason as interned in the store.
        """
        me = self.me.get(ko, None)
        if me != None: return me  #  Only do this once
        me = self.meIn(ko)
        if diag.chatty_flag>49: progress("CommandLine reason=%s ko=%s"%(self,me))
        ko.add(subj=me, pred=rdf.type, obj=reason.CommandLine, why=dontAsk)
        ko.add(subj=me, pred=reason.args, obj=self._string, why=dontAsk)
        return me
    
class BecauseOfExperience(Because):
    """Becase of the experience of this agent, as described in the string"""
    pass
    
class BecauseBuiltIn(Reason):
    """Because the built-in function given concluded so.
    A nested reason for running the function must be given"""
    def __init__(self, context, subj, pred, obj):
        Reason.__init__(self)
        self._subject = subj
        self._predicate = pred
        self._object = obj
        self._context = context
        
    def explain(self, ko, flags):
        "This is just a plain fact - or was at the time."
        me = self.me.get(ko, None)
        if me != None: return me  #  Only do this once
        me = self.meIn(ko)
        if diag.chatty_flag>49: progress("Fact reason=%s ko=%s"%(self,me))
        fact = ko.newFormula()
        fact.add(subj=subFormulaStandIn(ko, self._subject, flags=flags),
                 pred=self._predicate,
                 obj=subFormulaStandIn(ko, self._object, flags=flags),
                                                        why=dontAsk)
        fact = fact.close()
        ko.add(me, rdf.type, reason.Fact, why=dontAsk)
        standIn = formulaStandIn(ko,fact, flags=flags)
        ko.add(me, reason.gives, standIn, why=dontAsk)
##      if (1==0) and False and 'g' in flags:
##          for x in self._subject, self._object:
##              proofs = proofsOf.get(x, None)
##              if proofs != None:
##                  ko.add(me, reason.proof, proofs[0].explain(ko, flags=flags), why=dontAsk)

#       if self._proof != None:
#           ko.add(me, reason.proof, self._proof.explain(ko), why=dontAsk)
        return me

class BecauseIncludes(BecauseBuiltIn):
    """Because of the speific built-in log:includes"""
    pass

class BecauseSupports(BecauseBuiltIn):
    """Because of the very special build-in log:supports"""
    def __init__(self, context, subj, conclusion, pred, obj, reason):
        BecauseBuiltIn.__init__(self, context, subj, pred, obj)
        self.reason = []
        for statement in reason:
            if isinstance(statement, Reason):
                if isinstance(statement, BecauseBuiltIn):
                    if statement._context is conclusion:
                        self.reason.append(statement)
                        progress(statement)
                        progress("Is anybody home?")
            else:
                if statement.context() is conclusion:
                    self.reason.append(statement)
        self.conclusion = conclusion
        assert isTopLevel(conclusion)

    def __repr__(self):
        return '%s(%s,%s,%s,%s,%s,%s)' % (self.__class__.__name__,
                                             self._context,
                                             self._subject,
                                             self.conclusion,
                                             self._predicate,
                                             self._object,
                                             self.reason
                                    )

    def explain(self, ko, flags):
        "This is just a plain fact - or was at the time."
        me = self.me.get(ko, None)
        if me != None: return me  #  Only do this once
        me = self.meIn(ko)
        if diag.chatty_flag>49: progress("Fact reason=%s ko=%s"%(self,me))
        fact = ko.newFormula()
        m = ko.newBlankNode(why= dontAsk)
#        raise RuntimeError(m)
        fact.add(subj=self._subject, pred=self._predicate, obj=self._object,
                                                        why=dontAsk)
        fact = fact.close()
        ko.add(me, rdf.type, reason.Conclusion, why=dontAsk)
        standIn = formulaStandIn(ko,fact, flags=flags)
        ko.add(me, reason.gives, standIn, why=dontAsk)
        ko.add(subj=m, pred=rdf.type, obj=reason.Conjunction, why=dontAsk) 
        ko.add(me, reason.because, m, why=dontAsk)
        statementsForReason = {}  # reverse index: group by reason
        for s in self.reason:
            try:
                rea = getStatementReason(s)
            except:
                print s
                raise
            x = statementsForReason.get(rea, None)
            if x is None: statementsForReason[rea] = [s]
            else: x.append(s)
        for e in statementsForReason:
            r1 = e.explain(ko, flags=flags)
            ko.add(m, reason.component, r1, why=dontAsk)
##        if 'g' in flags:
##            for x in self._subject, self._object:
##                proofs = proofsOf.get(x, None)
##                if proofs != None:
##                    ko.add(me, reason.proof, proof[0].explain(ko, flags=flags), why=dontAsk)

#       if self._proof != None:
#           ko.add(me, reason.proof, self._proof.explain(ko), why=dontAsk)
        return me


# ends
