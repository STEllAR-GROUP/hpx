""" Query for cwm architecture

2003-09-07 split off from llyn.py
"""

QL_NS = "http://www.w3.org/2004/ql#"
from sparql2cwm import SPARQL_NS

from set_importer import Set, ImmutableSet, sorted

from RDFSink import Logic_NS, RDFSink, forSomeSym, forAllSym
from RDFSink import CONTEXT, PRED, SUBJ, OBJ, PARTS, ALL4
from RDFSink import N3_nil, N3_first, N3_rest, OWL_NS, N3_Empty, N3_List, List_NS
from RDFSink import RDF_NS_URI

from OrderedSequence import intersection, minus, indentString

import diag
from diag import chatty_flag, tracking, progress
from term import BuiltIn, LightBuiltIn, RDFBuiltIn, ArgumentNotLiteral, \
    HeavyBuiltIn, Function, ReverseFunction, MultipleFunction, \
    MultipleReverseFunction, UnknownType, Env, \
    Literal, Symbol, Fragment, FragmentNil,  Term, \
    CompoundTerm, List, EmptyList, NonEmptyList, ErrorFlag
from formula import StoredStatement, Formula
from why import Because, BecauseBuiltIn, BecauseOfRule, \
    BecauseOfExperience, becauseSubexpression, Reason, \
    BecauseSupports, BecauseMerge ,report, Premise, newTopLevelFormula, isTopLevel


BuiltinFeedError = (ArgumentNotLiteral, UnknownType)

import types
import sys
# from sets import Set  # only in python 2.3 and following
                        # set_importer does a cleaner job

INFINITY = 1000000000           # @@ larger than any number occurences


# State values as follows, high value=try first:
S_UNKNOWN =     99  # State unknown - to be [re]calculated by setup.
S_DONE =        80  # Exhausted all possible ways to saitsfy this. return now.
S_LIGHT_UNS_READY= 70  # Light, not searched yet, but can run
S_LIGHT_GO =    65  # Light, can run  Do this!
S_NOT_LIGHT =   60  # Not a light built-in, haven't searched yet.
S_LIGHT_EARLY=  50  # Light built-in, not ready to calculate, not searched yet.
S_NEED_DEEP=    45  # Can't search because of unbound compound term,
                    #   could do recursive unification
S_HEAVY_READY=  40  # Heavy built-in, search done,
                    #    but formula now has no vars left. Ready to run.
S_LIGHT_WAIT=   30  # Light built-in, not enough constants to calculate, search done.
S_HEAVY_WAIT=   20  # Heavy built-in, too many variables in args to calculate, search done.
S_REMOTE =      10  # Waiting for local query to be resolved as much as possible
#S_SATISFIED =   0  # Item has been staisfied, and is no longer a constraint, continue with others

stateName = { 
    S_UNKNOWN : "????",
    S_DONE :        "DONE",
    S_LIGHT_UNS_READY : "LtUsGo",
    S_LIGHT_GO : "LtGo",
    S_NOT_LIGHT : "NotLt",
    S_LIGHT_EARLY : "LtEarly",
    S_NEED_DEEP :  "Deep",
    S_HEAVY_READY :   "HvGo",
    S_LIGHT_WAIT : "LtWait",
    S_HEAVY_WAIT : "HvWait",
    S_REMOTE :   "Remote"}
#    S_SATISFIED:          "Satis" }





def think(knowledgeBase, ruleFormula=None, mode="", why=None):
    """Forward-chaining inference
    
    In the case in which rules are added back into the
    store. The store is used for read (normally canonical) and write
    (normally open) at the samne time.  It in fact has to be open.
    """
    if ruleFormula == None:
        ruleFormula = knowledgeBase
    assert knowledgeBase.canonical == None , "Must be open to add stuff:"+ `knowledgeBase `

    if diag.chatty_flag > 45: progress("think: rules from %s added to %s" %(
                                        knowledgeBase, ruleFormula))
    return InferenceTask(knowledgeBase, ruleFormula, mode=mode, why=why, repeat=1).run()

def applyRules(
                workingContext,    # Data we assume 
                ruleFormula = None,    # Where to find the rules
                targetContext = None):   # Where to put the conclusions
    """Once"""
    t = InferenceTask(workingContext, ruleFormula, targetContext)
    result = t.run()
    del(t)
    return result

def applyQueries(
                workingContext,    # Data we assume 
                ruleFormula = None,    # Where to find the rules
                targetContext = None):   # Where to put the conclusions
    """Once, nothing recusive, for a N3QL query"""
    t = InferenceTask(workingContext, ruleFormula, targetContext)
    t.gatherQueries(t.ruleFormula)
    result = t.runSmart()
    del(t)
    return result


def applySparqlQueries(
                workingContext,    # Data we assume 
                ruleFormula = None,    # Where to find the rules
                targetContext = None):   # Where to put the conclusions
    """Once, nothing recusive, for a N3QL query"""
    t = InferenceTask(workingContext, ruleFormula, targetContext, mode="q")
    t.gatherSparqlQueries(t.ruleFormula)
    result = t.runSmart()
    del(t)
    return result

class InferenceTask:
    """A task of applying rules or filters to information"""
    def __init__(self,
                workingContext,    # Data we assume 
                ruleFormula = None,    # Where to find the rules
                targetContext = None,   # Where to put the conclusions
                universals = Set(),        # Inherited from higher contexts
                mode="",                # modus operandi
                why=None,                       # Trace reason for all this
                repeat = 0):            # do it until finished
        """ Apply rules in one context to the same or another
    
        A rule here is defined by log:implies, which associates the template
        (aka premise, precondidtion, antecedent, body) to the conclusion
        (aka postcondition, head).
        """
        if diag.chatty_flag >20:
            progress("New Inference task, rules from %s" % ruleFormula)
        if targetContext is None: targetContext = workingContext # return new data to store
        if ruleFormula is None: self.ruleFormula = workingContext # apply own rules
        else: self.ruleFormula = ruleFormula
        self.ruleFor = {}
        self.hasMetaRule = 0

        self.workingContext, self.targetContext, self.mode, self.repeat = \
            workingContext, targetContext, mode, repeat
        self.store = self.workingContext.store

    def runSmart(self):
        """Run the rules by mapping rule interactions first"""
        rules= self.ruleFor.values()
        if self.targetContext is self.workingContext: #otherwise, there cannot be loops
            for r1 in rules:
                vars1 = r1.templateExistentials | r1.variablesUsed
                for r2 in rules:
                    vars2 = r2.templateExistentials | r2.variablesUsed
                    for s1 in r1.conclusion.statements:
                        s2 = None
                        p = None        
                        for s2 in r2.template.statements:
                            for p in PRED, SUBJ, OBJ:
                                if ((s1[p] not in vars1
                                        and not isinstance(s1[p], CompoundTerm))
                                    and (s2[p] not in vars2
                                        and not isinstance(s2[p], CompoundTerm))
                                    and (s1[p] is not s2[p])):
                                        break
                            else:
                                r1.affects[r2] = 1 # save transfer binding here?
                                if diag.chatty_flag > 20: progress(
                                    "%s can affect %s because %s can trigger %s" %
                                                (`r1`, `r2`, `s1`, `s2`))
                                break # can affect

                        else:  # that statement couldn't but
                            if diag.chatty_flag > 96:
                                progress("...couldn't beccause of ",s1,s2,p)
                            continue # try next one
                        break # can

        # Calculate transitive closure of "affects"
        for r1 in rules:
            r1.traceForward(r1)  # could lazy eval

        for r1 in rules:
            r1.indirectlyAffects.sort()
            r1.indirectlyAffectedBy.sort()
            
        # Print the affects matrix
        if diag.chatty_flag > 30:
            str = "%4s:" % "Aff"
            for r2 in rules:
                str+= "%4s" % `r2`
            progress(str)
            for r1 in rules:
                str= "%4s:" % `r1`
                for r2 in rules:
                    if r2.affects.get(r1, 0): str+= "%4s" % "X"
                    elif r1 in r2.indirectlyAffects: str +="%4s" % "-"
                    else: str+= "%4s" % " "
                progress(str)
            
        # Find cyclic subsystems
        # These are connected sets within which you can get from any node back to itself
        pool = rules[:]
        pool.sort()  # Sorted list, can use merge, intersection, etc
        cyclics = []
        while pool:
            r1 = pool[0]
            if r1 not in r1.indirectlyAffects:
                cyclic = [ r1 ]
                pool.remove(r1)
            else:
                if diag.chatty_flag > 90:
                    progress("%s indirectly affects %s and is affected by %s" %
                        (r1, r1.indirectlyAffects, r1.indirectlyAffectedBy))
                cyclic = intersection(r1.indirectlyAffects, r1.indirectlyAffectedBy)
                pool = minus(pool, cyclic)
            cyclics.append(CyclicSetOfRules(cyclic))
            if diag.chatty_flag > 90:
                progress("New cyclic: %s" % cyclics[-1])
            

        # Because the cyclic subsystems contain any loops, we know
        # there are no cycles between them. Therefore, there is a partial
        # ordering.  If we order them in that way, then we can resolve each
        # one just once without violating

        pool = cyclics[:]
        seq = []
        while pool:
            cy = pool[0] #  pick random one
            seq = partialOrdered(cy, pool) + seq

        if diag.chatty_flag > 10:
            progress("In order:" + `seq`)

        # TEMPORARY: test that
        n = len(seq)
        for i in range(n):
            for j in range(i-1):
                if compareCyclics(seq[i], seq[j]) < 0:
                    raise RuntimeError("Should not be:  %s < %s" %(seq[i], seq[j]))

        # Run the rules
        total = 0
        for cy in seq:
            total += cy.run()
        if diag.chatty_flag > 9:
            progress("Grand total %i facts found" % total)
        return total


    def run(self):
        """Perform task.
        Return number of  new facts"""
        self.gatherRules(self.ruleFormula)
        if self.hasMetaRule or not self.repeat:
            return self.runLaborious()
        return self.runSmart()

    def runLaborious(self):
        """Perform task.
        Return number of  new facts.
        Start again if new rule mayhave been generated."""
        grandtotal = 0
        iterations = 0
        self.ruleFor = {}
        needToCheckForRules = 1
        while 1:
            if needToCheckForRules:
                self.gatherRules(self.ruleFormula)
                needToCheckForRules = 0
            _total = 0
            iterations = iterations + 1
            for rule in self.ruleFor.values():
                found = rule.once()
                if (diag.chatty_flag >50):
                    progress( "Laborious: Found %i new stmts on for rule %s" %
                                                            (found, rule))
                _total = _total+found
                if found and (rule.meta or self.targetContext._closureMode):
                    needToCheckForRules = 1
            if diag.chatty_flag > 5: progress(
                "Total of %i new statements on iteration %i." %
                    (_total, iterations))
            if _total == 0: break
            grandtotal= grandtotal + _total
            if not self.repeat: break
        if diag.chatty_flag > 5: progress(
                "Grand total of %i new statements in %i iterations." %
                    (grandtotal, iterations))
        return grandtotal


    def gatherRules(self, ruleFormula):
        universals = Set() # @@ self.universals??
        for s in ruleFormula.statementsMatching(pred=self.store.implies):
            r = self.ruleFor.get(s, None)
            if r != None: continue
            con, pred, subj, obj  = s.quad
            if (isinstance(subj, Formula)
                and isinstance(obj, Formula)):
                v2 = universals | ruleFormula.universals() # Note new variables can be generated
                r = Rule(self, antecedent=subj, consequent=obj, statement=s,  variables=v2)
                self.ruleFor[s] = r
                if r.meta: self.hasMetaRule = 1
                if (diag.chatty_flag >30):
                    progress( "Found rule %r for statement %s " % (r, s))

        for F in ruleFormula.each(pred=self.store.implies, obj=self.store.Truth): #@@ take out when --closure=T ??
            self.gatherRules(F)  #See test/rules13.n3, test/schema-rules.n3 etc

    def gatherQueries(self, ruleFormula):
        "Find a set of rules in N3QL"
        universals = Set() # @@ self.universals??
        ql_select = self.store.newSymbol(QL_NS + "select")
        ql_where = self.store.newSymbol(QL_NS + "where")
        for s in ruleFormula.statementsMatching(pred=ql_select):
            r = self.ruleFor.get(s, None)
            if r != None: continue
            con, pred, query, selectClause  = s.quad
            whereClause= ruleFormula.the(subj=query, pred=ql_where)
            if whereClause == None: continue # ignore (warning?)
            
            if (isinstance(selectClause, Formula)
                and isinstance(whereClause, Formula)):
                v2 = universals | ruleFormula.universals() # Note new variables can be generated
                r = Rule(self, antecedent=whereClause, consequent=selectClause,
                                statement=s,  variables=v2)
                self.ruleFor[s] = r
                if r.meta: self.hasMetaRule = 1
                if (diag.chatty_flag >30):
                    progress( "Found rule %r for statement %s " % (r, s))

    def gatherSparqlQueries(self, ruleFormula):
        "Find the rules in SPARQL"
        store = self.store
        sparql = store.newSymbol(SPARQL_NS)
        
        for from_statement in ruleFormula.statementsMatching(pred=sparql['data']):
            working_context_stand_in = from_statement.object()
            ruleFormula = ruleFormula.substitution({working_context_stand_in: self.workingContext})
        query_root = ruleFormula.any(pred=store.type, obj=sparql['ConstructQuery'])
        if not query_root:
            # This is wrong
            query_root = ruleFormula.any(pred=store.type, obj=sparql['SelectQuery'])
        if not query_root:
            query_root = ruleFormula.any(pred=store.type, obj=sparql['AskQuery'])
        # query_root is a very boring bNode
        if query_root:
            #construct query
            for where_triple in ruleFormula.statementsMatching(subj=query_root, pred=sparql['where']):
                where_clause = where_triple.object()
                #where_clause is the tail of the rule
                implies_clause = ruleFormula.the(subj=where_clause, pred=store.implies)
                assert implies_clause is not None, ("where=%s, f=%s" % (where_clause.debugString(), ruleFormula.debugString()))
                #implies_clause is the head of the rule
                v2 = ruleFormula.universals().copy()
                r = Rule(self, antecedent=where_clause, consequent=implies_clause,
                         statement=where_triple, variables=v2)
                self.ruleFor[where_triple] = r
                if r.meta: self.hasMetaRule = 1
                if (diag.chatty_flag >30):
                    progress( "Found rule %r for statement %s " % (r, where_triple))


def partialOrdered(cy1, pool):
    """Return sequence conforming to the partially order in a set of cyclic subsystems
    
    Basially, we find the dependencies of a node and remove them from the pool.
    Then, any node in the pool can be done earlier, because has no depndency from those done.
    """
    seq = []
    for r1 in cy1:   # @@ did just chose first rule cy[0], but didn't get all
        for r2 in  r1.affects:
            if r2 not in cy1:  # An external dependency
                cy2 = r2.cycle
                if cy2 in pool:
                    seq = partialOrdered(cy2, pool) + seq
    pool.remove(cy1)
    if diag.chatty_flag > 90: progress("partial topo: %s" % `[cy1] + seq`)
    return [cy1] + seq

class CyclicSetOfRules:
    """A set of rules which are connected
    """
    def __init__(self, rules):
        self.rules = rules
        for r1 in rules:
            r1.cycle = self

    def __getitem__(self, i):
        return self.rules[i]

    def __repr__(self):
        return `self.rules`

    def run(self):
        "Run a cyclic subset of the rules"
        if diag.chatty_flag > 20:
            progress()
            progress("Running cyclic system %s" % (self))
        if len(self.rules) == 1:
            rule = self.rules[0]
            if not rule.affects.get(rule, 0):
#               rule.already = None # Suppress recording of previous answers
                # - no, needed to remove dup bnodes as in test/includes/quant-implies.n3 --think
                # When Rule.once is smarter about not iterating over things not mentioned elsewhere,
                # can remove this.
                return rule.once()
                
        agenda = self.rules[:]
        total = 0
        for r1 in self.rules:
            af = r1.affects.keys()
            af.sort()
            r1.affectsInCyclic = intersection(self.rules, af)
        while agenda:
            rule = agenda[0]
            agenda = agenda[1:]
            found = rule.once()
            if diag.chatty_flag > 20: progress("Rule %s gave %i. Affects:%s." %(
                        rule, found, rule.affectsInCyclic))
            if found:
                total = total + found
                for r2 in rule.affectsInCyclic:
                    if r2 not in agenda:
                        if diag.chatty_flag > 30: progress("...rescheduling", r2)
                        agenda.append(r2)
        if diag.chatty_flag > 20: progress("Cyclic subsystem exhausted")
        return total
        

def buildPattern(workingContext, template):
    """Make a list of unmatched statements including special
    builtins to check something is universally quantified"""
    unmatched = template.statements[:]
    for v in template.occurringIn(template.universals()):
        if diag.chatty_flag > 100: progress(
            "Template %s has universalVariableName %s, formula is %s" % (template, v, template.debugString()))
        unmatched.append(StoredStatement((workingContext,
                template.store.universalVariableName,
                workingContext,
                v
#               workingContext.store.newLiteral(v.uriref())
                                          )))
    return unmatched

def buildStrictPattern(workingContext, template):
    unmatched = buildPattern(workingContext, template)
    for v in template.occurringIn(template.existentials()):
        if diag.chatty_flag > 100: progress(
            "Tempate %s has existentialVariableName %s, formula is %s" % (template, v, template.debugString()))
        unmatched.append(StoredStatement((workingContext,
                template.store.existentialVariableName,
                workingContext,
                v
#               workingContext.store.newLiteral(v.uriref())
                                          )))
##    for v in template.variables():
##      if diag.chatty_flag > 100: progress(
##          "Tempate %s has enforceUniqueBinding %s, formula is %s" % (template, v, template.debugString()))
##      unmatched.append(StoredStatement((workingContext,
##              template.store.enforceUniqueBinding,
##                v,
##              workingContext.store.newLiteral(v.uriref())
##                                          )))
    return unmatched
    
nextRule = 0
class Rule:

    def __init__(self, task, antecedent, consequent, statement, variables):
        """Try a rule
        
        Beware lists are corrupted. Already list is updated if present.
        The idea is that, for a rule which may be tried many times, the constant 
        processing is done in this rather than in Query().
        
        The already dictionary is used to track bindings.
        This less useful when not repeating (as in --filter), but as in fact
        there may be several ways in which one cane get the same bindings,
        even without a repeat.
        """
        global nextRule
        self.task = task
        self.template = antecedent
        self.conclusion = consequent
        self.store = self.template.store
        self.statement = statement      #  original statement
        self.number = nextRule = nextRule+1
        self.meta = self.conclusion.contains(pred=self.conclusion.store.implies) #generate rules?
#       if task.repeat: self.already = []    # No neat to track dups if not 
#       else: self.already = None
        self.already = []
        self.affects = {}
        self.indirectlyAffects = []
        self.indirectlyAffectedBy = []
        self.affectsInCyclic = []
        self.cycle = None
        
        # When the template refers to itself, the thing we are
        # are looking for will refer to the context we are searching
        # Similarly, references to the working context have to be moved into the
        # target context when the conclusion is drawn.
    
    
#       if self.template.universals() != Set():
#           raise RuntimeError("""Cannot query for universally quantified things.
#           As of 2003/07/28 forAll x ...x cannot be on left hand side of rule.
#           This/these were: %s\n""" % self.template.universals())
    
        self.unmatched = buildPattern(task.workingContext, self.template)
        self.templateExistentials = self.template.existentials().copy()
        _substitute({self.template: task.workingContext}, self.unmatched)
    
        variablesMentioned = self.template.occurringIn(variables)
        self.variablesUsed = self.conclusion.occurringIn(variablesMentioned)
        for x in variablesMentioned:
            if x not in self.variablesUsed:
                self.templateExistentials.add(x)
        if diag.chatty_flag >20:
            progress("New Rule %s ============ looking for:" % `self` )
            for s in self.template.statements: progress("    ", `s`)
            progress("=>")
            for s in self.conclusion.statements: progress("    ", `s`)
            progress("Universals declared in outer " + seqToString(variables))
            progress(" mentioned in template       " + seqToString(variablesMentioned))
            progress(" also used in conclusion     " + seqToString(self.variablesUsed))
            progress("Existentials in template     " + seqToString(self.templateExistentials))
        return

    def once(self):
        if diag.chatty_flag >20:
            progress("Trying rule %s ===================" % self )
            progress( setToString(self.unmatched))
        task = self.task
        query = Query(self.store,
                        unmatched = self.unmatched[:],
                        template = self.template,
                        variables = self.variablesUsed.copy(),
                        existentials = self.templateExistentials.copy(),
                        workingContext = task.workingContext,
                        conclusion = self.conclusion,
                        targetContext = task.targetContext,
                        already = self.already,
                      ###
                        rule = self.statement,
                      ###
                        interpretBuiltins = 1,    # (...)
                        meta = task.workingContext,
                        mode = task.mode)
        Formula.resetRenames()
        total = query.resolve()
        Formula.resetRenames(False)
        if diag.chatty_flag > 20:
            progress("Rule try generated %i new statements" % total)
        return total
    
    def __repr__(self):
        if self in self.affects: return "R"+`self.number`+ "*"
        return "R"+`self.number`

    def compareByAffects(other):
        if other in self.indirectlyAffects: return -1  # Do me earlier
        if other in self.indirectlyAffectedBy: return 1
        return 0
        

    def traceForward(self, r1):
        for r2 in r1.affects:
            if r2 not in self.indirectlyAffects:
                self.indirectlyAffects.append(r2)
                r2.indirectlyAffectedBy.append(self)
                self.traceForward(r2)
#           else:
#               self.__setattr__("leadsToCycle", 1)
    
def testIncludes(f, g, _variables=Set(),  bindings={}, interpretBuiltins = 0):
    """Return whether or nor f contains a top-level formula equvalent to g.
    Just a test: no bindings returned."""
    if diag.chatty_flag >30: progress("testIncludes ============\nseeing if %s entails %s" % (f, g))
#    raise RuntimeError()
    if not(isinstance(f, Formula) and isinstance(g, Formula)): return 0

    assert f.canonical is f, f.debugString()
    assert g.canonical is g
    m = diag.chatty_flag
    diag.chatty_flag = 0
    if m > 60: progress("Before rename: ", f.debugString())
    f = f.renameVars()
    if m > 60: progress("After rename: ", f.debugString())
    diag.chatty_flag = m
    if diag.chatty_flag >100: progress("Formula we are searching in is\n%s" % g.debugString())
    unmatched = buildPattern(f, g)
    templateExistentials = g.existentials()
    more_variables = g.universals().copy()
    _substitute({g: f}, unmatched)
    
#    if g.universals() != Set():
#       raise RuntimeError("""Cannot query for universally quantified things.
#       As of 2003/07/28 forAll x ...x cannot be on left hand side of rule.
#       This/these were: %s\n""" % g.universals())

#    if bindings != {}: _substitute(bindings, unmatched)

    if diag.chatty_flag > 20:
        progress( "# testIncludes BUILTIN, %i terms in template %s, %i unmatched, %i template variables" % (
            len(g.statements),
            `g`[-8:], len(unmatched), len(templateExistentials)))
        if diag.chatty_flag > 80:
            for v in _variables:
                progress( "    Variable: " + `v`[-8:])

    result = Query(f.store,
                unmatched=unmatched,
                template = g,
                variables=Set(),
                interpretBuiltins = interpretBuiltins,
                existentials=_variables | templateExistentials | more_variables,
                justOne=1, mode="").resolve()

    if diag.chatty_flag >30: progress("=================== end testIncludes =" + `result`)
    return result


def n3Equivalent(g, f, env1, env2, vars=Set([]),
                 universals=Set(), existentials=Set([]),
                 n1Source=42, n2Source=42):    
    """Return whether or nor f contains a top-level formula equvalent to g.
    Just a test: no bindings returned."""
    if diag.chatty_flag >30: progress("Query.py n3Equivalent ============\nseeing if %s equals %s" % (f, g))
#    raise RuntimeError()
    if not(isinstance(f, Formula) and isinstance(g, Formula)): pass
    elif f is g:
        yield env1, env2
    elif len(f) > len(g):
        pass
    else:

        assert f.canonical is f, "%s, %s" % (f.debugString(), f.canonical.debugString())
        assert g.canonical is g, "%s, %s" % (g.debugString(), g.canonical.debugString())

        m = diag.chatty_flag
        diag.chatty_flag = 0
        if m > 60: progress("Before rename: ", f.debugString())
        f = f.renameVars()
        if m > 60: progress("After rename: ", f.debugString())
        diag.chatty_flag = m
        unmatched = buildStrictPattern(f, g)
        templateExistentials = g.existentials() | g.universals() | existentials
        more_variables = Set(vars)
        _substitute({g: f}, unmatched)
        
        if env1: _substitute(env1.asDict(), unmatched)

        if diag.chatty_flag > 20:
            progress( "# testEqual BUILTIN, %i terms in template %s, %i unmatched, %i template variables" % (
                len(g.statements),
                `g`[-8:], len(unmatched), len(templateExistentials)))
            if diag.chatty_flag > 80:
                for v in vars:
                    progress( "    Variable: " + `v`[-8:])

        result = Query(f.store,
                    unmatched=unmatched,
                    template = g,
                    variables=more_variables,
                    workingContext = f,
                    interpretBuiltins = False,
                    existentials= templateExistentials ,
                    justReturn=1, mode="").resolve()

        if diag.chatty_flag >30: progress("=================== end n3Equivalent =" + `result`)
        if not result: result = []
        for x in result:
            for k, (v, source) in x.items():
                env1 = env1.bind(k, (v, env2.id))
            yield env1, env2
##    return [(x, None) for x in result]



############################################################## Query engine
#
# Template matching in a graph
#
# Optimizations we have NOT done:
#   - storing the tree of bindings so that we don't have to duplicate them another time
#   - using that tree to check only whether new data would extend it (very cool - memory?)
#      (this links to dynamic data, live variables.)
#   - recognising in advance disjoint graph templates, doing cross product of separate searches
#
# Built-Ins:
#   The trick seems to be to figure out which built-ins are going to be faster to
# calculate, and so should be resolved before query terms involving a search, and
# which, like those involving recursive queries or net access, will be slower than a query term,
# and so should be left till last.
#   I feel that it should be possible to argue about built-ins just like anything else,
# so we do not exclude these things from the query system. We therefore may have both light and
# heavy built-ins which still have too many variables to calculate at this stage.
# When we do the variable substitution for new bindings, these can be reconsidered.


class Queue([].__class__):
    __slots__ = ['statements', 'bNodes']
    list = [].__class__

    def __init__(self, other=[], metaSource = None):
        self.list.__init__(self, other)
        if isinstance(metaSource, Queue):
            for k in self.__slots__:
                setattr(self, k, getattr(metaSource, k).copy())
        else:
            self.statements = Set()
            self.bNodes = Set()
            pass #fill in slots here

    def popBest(self):
        best = len(self) -1 # , say...
        i = best - 1
        while i >=0:
            if (self[i].state > self[best].state
                or (self[i].state == self[best].state
                    and self[i].short < self[best].short)): best=i
            i = i - 1                
        item = self[best]
        self.remove(item)
        return item

    def __repr__(self):
        return 'Queue(%s, bNodes=%s)' % (list.__repr__(self), self.bNodes)

#Queue = [].__class__

class Chain_Step(object):
    def __init__(self, vars, existentials, queue, env, parent=None, evidence=[]):
        self.vars = vars
        self.existentials = existentials
        self.lines = queue
        self.env = env
        assert parent is None
        self.parent = parent
        self.evidence = evidence

    def popBest(self):
        return self.lines.popBest()

    def copy(self):
        retVal = self.__class__(self.vars, self.existentials, self.lines, self.env, self.parent, self.evidence)
        return retVal

    def done(self):
        return not self.lines

    def __cmp__(self, other):
        return cmp(len(other.lines), len(self.lines))

    def __repr__(self):
        return "%s(lines=%r,\n\tenv=%r,\n\tparent=%r,\n\tevidence=%r)" % (self.__class__.__name__, self.lines, self.env, self.parent, self.evidence)


def returnWrapper(f):
    def g(*args, **keywords):
        retVal = f(*args, **keywords)
        progress('%s() returns %s' % (f.func_name, retVal))
        return retVal
    return g

class Query(Formula):
    """A query holds a hypothesis/antecedent/template which is being matched aginst (unified with)
    the knowledge base."""
    def __init__(self,
               store,
               unmatched=[],           # Tuple of interned quads we are trying to match CORRUPTED
               template = None,         # Actually, must have one
               variables=Set(),           # List of variables to match and return CORRUPTED
               existentials=Set(),        # List of variables to match to anything
                                    # Existentials or any kind of variable in subexpression
               workingContext = None,
               conclusion = None,
               targetContext = None,
               already = None,      # Dictionary of matches already found
               rule = None,                 # The rule statement
               interpretBuiltins = 0,        # List of contexts in which to use builtins
               justOne = 0,         # Flag: Stop when you find the first one
               justReturn = 0,      # Flag: Return bindings, don't conclude
               mode = "",           # Character flags modifying modus operandi
            meta = None):           # Context to check for useful info eg remote stuff

        
        if diag.chatty_flag > 50:
            progress( "Query: created with %i terms. (justone=%i, wc=%s)" % 
                    (len(unmatched), justOne, workingContext))
            if diag.chatty_flag > 80: progress( setToString(unmatched))
            if diag.chatty_flag > 90 and interpretBuiltins: progress(
                "iBuiltIns=1 ")

        Formula.__init__(self, store)
        self.statements = Queue()   #  Unmatched with more info
#       self.store = store      # Initialized by Formula
        self.variables = variables
        self._existentialVariables = existentials
        self.workingContext = workingContext
        self.conclusion = conclusion
        self.targetContext = targetContext
        self.justOne = justOne
        self.already = already
        self.rule = rule
        self.template = template  # For looking for lists
        self.meta = meta
        self.mode = mode
        self.lastCheckedNumberOfRedirections = 0
        self.bindingList = []
        self.justReturn = justReturn
        realMatchCount = 0
        if justReturn and not variables:
            self.justOne = True
        for quad in unmatched:
            item = QueryItem(self, quad)
            if not item.setup(allvars=variables|existentials, unmatched=unmatched,
                        interpretBuiltins=interpretBuiltins, mode=mode):
                if diag.chatty_flag > 80: progress(
                                    "match: abandoned, no way for "+`item`)
                self.noWay = 1
                return  # save time
            if not item.builtIn:
                realMatchCount += 1    
            self.statements.append(item)
        if justReturn and realMatchCount > len(workingContext):
            self.noWay = 1
            return
        return
        
    def resolve(self):
        if hasattr(self, "noWay"): return 0
        k = self.matchFormula(self.statements, self.variables, self._existentialVariables)
        if self.justReturn:
            return self.bindingList
        return k

    def checkRedirectsInAlready(self):
        """Kludge"""
        n = len(self.targetContext._redirections)
        if n  > self.lastCheckedNumberOfRedirections:
            self.lastCheckedNumberOfRedirections = n
            self.redirect(self.targetContext._redirections)
            
    def redirect(self, redirections):
        for bindings in self.already:
            for var, value in bindings.items():
                try:
                    x = redirections[value]
                except:
                    pass
                else:
                    if diag.chatty_flag>29: progress("Redirecting binding %r to %r" % (value, x))
                    bindings[var] = x

    def conclude(self, bindings, evidence = [], extraBNodes = Set(), allBindings=None):
        """When a match found in a query, add conclusions to target formula.

        Returns the number of statements added."""
        if self.justOne:
            self.bindingList = [{}]
            return 1   # If only a test needed
        if self.justReturn:
            if bindings not in self.bindingList:
#                progress('CONCLUDE bindings = %s' % bindings)
                self.bindingList.append(bindings)
            return 1

        if diag.chatty_flag >60:
                        progress( "Concluding tentatively...%r" % bindings)
        if self.already != None:
            self.checkRedirectsInAlready() # @@@ KLUDGE - use delegation and notification systme instead
            if bindings in self.already:
                if diag.chatty_flag > 30:
                    progress("@@ Duplicate result: %r is in %r" %  (bindings, self.already))
                return 0
            if diag.chatty_flag > 30: progress("Not duplicate: %r" % bindings)
            self.already.append(bindings)
        else: 
            if diag.chatty_flag >60:
                        progress( "No duplication check")
        
        if diag.tracking:
            if allBindings is None:
                allBindings = bindings
            for loc in xrange(len(evidence)):
                r = evidence[loc]
                
                if isinstance(r, BecauseSupportsWill):                    
                    evidence[loc] = BecauseSupports(*([smarterSubstitution(k, allBindings,
                        r.args[1], why=Because("I support it: "), exception=[r.args[2]]) for k in r.args] +
                        [[k for k in evidence if isinstance(k, (StoredStatement, Reason))]]))
                if isinstance(r, BecauseBuiltInWill):
                    evidence[loc] = BecauseBuiltIn(*[smarterSubstitution(k, allBindings,
                        r.args[0], why=Because("I include it: " + k.debugString() + `allBindings`)) for k in r.args[1:]])
            reason = BecauseOfRule(self.rule, bindings=allBindings, knownExistentials = extraBNodes,
                            evidence=evidence, kb=self.workingContext)
#           progress("We have a reason for %s of %s with bindings %s" % (self.rule, reason, alBindings))
        else:
            reason = None

        es, exout = (self.workingContext.existentials() | extraBNodes), Set() #self.workingContext.existentials() | 
        for var, (val, source) in bindings.items():
            if isinstance(val, Exception):
                if "q" in self.mode: # How nice are we?
                    raise ValueError(val)
                return 0
            intersection = val.occurringIn(es) #  Take time for large number of bnodes?
            if intersection:   
                exout.update(intersection)
                if diag.chatty_flag > 25: progress(
                "Match found to that which is only an existential: %s -> %s" %
                                                    (var, val))
                for val2 in intersection:
                    if val not in self.targetContext.existentials():
                        if self.conclusion.occurringIn([var]):
                            self.targetContext.declareExistential(val2)

        # Variable renaming

        b2 = bindings.asDict()
#        b2[self.conclusion] = self.targetContext  # What does this mean?
        ok = self.targetContext.universals() 
        # It is actually ok to share universal variables with other stuff
        poss = self.conclusion.universals().copy()
        for x in poss.copy():
            if x in ok: poss.remove(x)
        poss_sorted = list(poss)
        poss_sorted.sort(Term.compareAnyTerm)
        #progress(poss)

#        vars = self.conclusion.existentials() + poss  # Terms with arbitrary identifiers
#        clashes = self.occurringIn(targetContext, vars)    Too slow to do every time; play safe
        if diag.chatty_flag > 25:
            s=""
        for v in poss_sorted:
            v2 = self.targetContext.newUniversal()
            b2[v] =v2   # Regenerate names to avoid clash
            if diag.chatty_flag > 25: s = s + ",uni %s -> %s" %(v, v2)
        for v in sorted(list(self.conclusion.existentials()), Term.compareAnyTerm):
            if v not in exout:
                v2 = self.targetContext.newBlankNode()
                b2[v] =v2   # Regenerate names to avoid clash
                if diag.chatty_flag > 25: s = s + ",exi %s -> %s" %(v, v2)
            else:
                if diag.chatty_flag > 25: s = s + (", (%s is existential in kb)"%v)
        if diag.chatty_flag > 25:
            progress("Variables regenerated: universal " + `poss`
                + " existential: " +`self.conclusion.existentials()` + s)
        

        if diag.chatty_flag>19:
            progress("Concluding DEFINITELY" + bindingsToString(b2) )
        before = self.store.size
        _, delta = self.targetContext.loadFormulaWithSubstitution(
                    self.conclusion, b2, why=reason, cannon=True)
        if diag.chatty_flag>29 and delta:
            progress(" --- because of: %s => %s, with bindings %s" % (self.template.debugString(),
                                                                      self.conclusion.debugString(),
                                                                      b2))
        if diag.chatty_flag> 40:
            progress("Added %i, nominal size of store changed from %i to %i."%(delta, before, self.store.size))
        return delta #  self.store.size - before


##################################################################################

    def matchFormula(query, queue, variables, existentials, env=Env()):
        total = 0
        stack = [Chain_Step(variables, existentials, queue, env)]
        while stack:
            if diag.chatty_flag > 150:
                progress(stack)
            workingStep = stack.pop()
            if not workingStep.done():
                queue = workingStep.lines
                evidence = workingStep.evidence
                bindings = workingStep.env
                variables = workingStep.vars
                existentials = workingStep.existentials
                
                item = workingStep.popBest()

                con, pred, subj, obj = item.quad
                state = item.state
                if state == S_DONE:  # After bindNew, could be undoable.
                    nbs = []
                elif state == S_LIGHT_UNS_READY:          # Search then 
                    nbs = item.tryBuiltin(queue, bindings, evidence=evidence)
                    item.state = S_LIGHT_EARLY   # Unsearched, try builtin @@@@@@@@@ <== need new state here
                elif state == S_LIGHT_GO:
                    nbs = item.tryBuiltin(queue, bindings, evidence=evidence)
                    item.state = S_DONE   # Searched.
                elif (state == S_LIGHT_EARLY or state == S_NOT_LIGHT or
                                        state == S_NEED_DEEP): #  Not searched yet
                    nbs = item.tryDeepSearch(queue, bindings)
                elif state == S_HEAVY_READY:  # not light, may be heavy; or heavy ready to run
                    if pred is query.store.includes: # and not diag.tracking:  # don't optimize when tracking?
                        variables = variables.copy()
                        existentials = existentials.copy()
                        nbs = item.doIncludes(queue, existentials, variables, bindings)
                    elif pred is query.store.supports:
                        variables = variables.copy()
                        existentials = existentials.copy()
                        nbs = item.doSupports(queue, existentials, variables, bindings)
                    else:
                        item.state = S_HEAVY_WAIT  # Assume can't resolve
                        nbs = item.tryBuiltin(queue, bindings, evidence=evidence)
                    item.state = S_DONE
                elif state == S_REMOTE: # Remote query -- need to find all of them for the same service
                    items = [item]
                    for i in queue[:]:
                        if i.state == S_REMOTE and i.service is item.service: #@@ optimize which group is done first!
                            items.append(i)
                            queue.remove(i)
                    nbs = query.remoteQuery(items)
                    item.state = S_DONE  # do not put back on list
                elif state ==S_HEAVY_WAIT or state == S_LIGHT_WAIT:
                    if item.quad[PRED] is query.store.universalVariableName or \
                       item.quad[PRED] is query.store.existentialVariableName:
                        ### We will never bind this variable in the first place
                        item.state = S_DONE
                        nbs = []
                    else:
                        if diag.chatty_flag > 20 :
                            progress("@@@@ Warning: query can't find term which will work.")
                            progress( "   state is %s, queue length %i" % (state, len(queue)+1))
                            progress("@@ Current item: %s" % `item`)
                            progress(queueToString(queue))
                        continue  # Forget it
                else:
                    raise RuntimeError, "Unknown state " + `state`


                stack_extent = []
                for nb, reason in nbs:
                    assert isinstance(nb, dict), nb
                    q2 = Queue([], queue)
                    if query.justReturn:
                        ### What does the following do?
                        ### If we are doing a 1::1 match, record everything we have matched
                        if isinstance(reason, StoredStatement):
                            if reason not in q2.statements and \
                               reason[CONTEXT] is query.workingContext:
                                q2.statements.add(reason)
                            else:
                                continue
                    if isinstance(reason, StoredStatement):
                        if True or reason[CONTEXT] is not query.workingContext:
                            for m in nb.values():
                                if isinstance(m, tuple):
                                    m = m[0]
                                if m in reason[CONTEXT].existentials():
                                    q2.bNodes.add(m)
                                    if diag.chatty_flag > 80:
                                        ### These get picked up from log:includes
                                        ### {[:b :c]} log:includes {?X :b :c} ...
                                        progress('Adding bNode %s, now %s' % (m, q2.bNodes))
                    new_thing = False
                    try:
                        new_env = bindings.flatten(nb)
                    except ValueError:
                        pass
                    else:
                        new_thing = True
                    for i in queue:
                        newItem = i.clone()
                        newItem.bindNew(new_env) ## Is this right? I was hoping to avoid this
                        q2.append(newItem)  #@@@@@@@@@@  If exactly 1 binding, loop (tail recurse)
                    if new_thing:
                        if diag.chatty_flag > 70:
                            progress("query.py bindings nb new_entry:", bindings, nb, new_env)
                        new_step = Chain_Step(variables, existentials, q2, new_env, workingStep.parent, workingStep.evidence + [reason])
                        stack_extent.append(new_step)

                if item.state != S_DONE:
                    queue.append(item)
                    new_step = workingStep.copy()
                    stack_extent.append(new_step)
                stack.extend(stack_extent)

                    

            else:
                if workingStep.parent is not None:
                    raise RuntimeError("We are not chaining yet.\n How did I get here?")
                else:
                    total = query.conclude(workingStep.env.filter(workingStep.vars), allBindings=workingStep.env
                                          , evidence=workingStep.evidence, extraBNodes = workingStep.lines.bNodes) + total  # No terms left .. success!
                    #raise RuntimeError("I need to conclude here, workingStep=%s" % workingStep)
                    pass
        return total


    def matchFormula2(query,
               queue,               # Set of items we are trying to match CORRUPTED
               variables,           # List of variables to match and return CORRUPTED
               existentials,        # List of variables to match to anything
                                    # Existentials or any kind of variable in subexpression
               bindings = Env(),       # Bindings discovered so far
               newBindings = Env(),    # New bindings not yet incorporated
               evidence = []):      # List of statements supporting the bindings so far
        """ Iterate on the remaining query items
    bindings      collected matches already found
    newBindings  matches found and not yet applied - used in recursion
    
    You probably really need the state diagram to understand this
    http://www.w3.org/2000/10/swap/doc/states.svg
    even if it is a bit out of date.
        """
        total = 0
        assert isinstance(bindings, Env)
        assert isinstance(newBindings, Env), 'env is an %s, not an %s' % (newBindings.__class__, Env)
        if diag.chatty_flag > 59:
            progress( "QUERY2: called %i terms, %i bindings %s, (new: %s)" %
                      (len(queue), len(bindings), `bindings`,
                       `newBindings`))
            if diag.chatty_flag > 90: progress( queueToString(queue))

        newBindingItems = newBindings.items()
        while newBindingItems:   # Take care of business left over from recursive call
            pair = newBindingItems.pop(0)
            if isinstance(pair[1], tuple):
                pair = (pair[0], pair[1][0])
            else:
                raise RuntimeError
            if diag.chatty_flag>95: progress("    new binding:  %s -> %s" % (`pair[0]`, `pair[1]`))
            if pair[0] in variables:
                variables.remove(pair[0])
                bindings = bindings.newBinding(pair[0], (pair[1], None))
            else:      # Formulae aren't needed as existentials, unlike lists. hmm.
                ### bindings.update({pair[0]: pair[1]})  # remove me!!!!!
#               if diag.tracking: raise RuntimeError(pair[0], pair[1])
                #bindings.update({pair[0]: pair[1]})  # Record for proof only
                if pair[0] not in existentials:
                    if isinstance(pair[0], List):
                        # pair[0] should be a variable, can't be a list, surely
                        del newBindings[pair[0]]
                        #We can accidently bind a list using (1 2 3) rdf:rest (?x 3).
                        #This finds the true binding
                        reallyNewBindingsList = pair[0].unify(
                                    pair[1], bindings, Env(), variables | existentials)
                        ## I'm a bit parenoid. If we did not find a binding ...
                        if not reallyNewBindingsList or not hasattr(
                                        reallyNewBindingsList, '__iter__'):
                            return 0
                        reallyNewBindingsList = [x for x in reallyNewBindingsList]
                        try:
                            reallyNewBindings = reallyNewBindingsList[0][0] #We don't deal
                            # with multiple ways to bind
                        except:
                            print 'we lost'
                            print pair[0], pair[1]
                            a = pair[0].unify(
                                    pair[1], variables, existentials, bindings)
                            print a
                            print a[0]
                            print a[0][0]
                            raise
                        newBindingItems.extend(reallyNewBindings.items())
                        newBindings = newBindings.update2(reallyNewBindings)
                    else:
                        if diag.chatty_flag > 40:  # Reasonable
                            progress("Not in existentials or variables but now bound:", `pair[0]`)
                elif diag.tracking: bindings = bindings.newBinding(pair[0], (pair[1], None))
                if not isinstance(pair[0], CompoundTerm) and ( # Hack - else rules13.n3 fails @@
                    pair[0] in existentials):    # Hack ... could be bnding from nested expression
                    existentials.remove(pair[0]) # Can't match anything anymore, need exact match

        # Perform the substitution, noting where lists become boundLists.
        # We do this carefully, messing up the order only of things we have already processed.
        if newBindings != {}:
            for item in queue:
                if item.bindNew(newBindings) == 0: return 0


        while len(queue) > 0:

            if (diag.chatty_flag > 90):
                progress( "query iterating with %i terms, %i bindings: %s; %i new bindings: %s ." %
                          (len(queue),
                           len(bindings),bindingsToString(bindings),
                           len(newBindings),bindingsToString(newBindings)))
                progress ( queueToString(queue))


            # Take best.  (Design choice: search here or keep queue in order)
            # item = queue.pop()
            item = queue.popBest()
            
            if diag.chatty_flag>49:
                progress( "Looking at " + `item`)
                progress( "...with vars("+seqToString(variables)+")"
                         + " ExQuVars:("+seqToString(existentials)+")")
            con, pred, subj, obj = item.quad
            state = item.state
            if state == S_DONE:  # After bindNew, could be undoable.
                return total # Forget it -- must be impossible
            if state == S_LIGHT_UNS_READY:              # Search then 
                nbs = item.tryBuiltin(queue, bindings, evidence=evidence)
                item.state = S_LIGHT_EARLY   # Unsearched, try builtin @@@@@@@@@ <== need new state here
            elif state == S_LIGHT_GO:
                nbs = item.tryBuiltin(queue, bindings, evidence=evidence)
                item.state = S_DONE   # Searched.
            elif (state == S_LIGHT_EARLY or state == S_NOT_LIGHT or
                                    state == S_NEED_DEEP): #  Not searched yet
                nbs = item.tryDeepSearch(queue, Env())
            elif state == S_HEAVY_READY:  # not light, may be heavy; or heavy ready to run
                if pred is query.store.includes: # and not diag.tracking:  # don't optimize when tracking?
                    nbs = item.doIncludes(queue, existentials, variables, bindings)
                elif pred is query.store.supports:
                    nbs = item.doSupports(queue, existentials, variables, bindings)
                else:
                    item.state = S_HEAVY_WAIT  # Assume can't resolve
                    nbs = item.tryBuiltin(queue, bindings, evidence=evidence)
                item.state = S_DONE
            elif state == S_REMOTE: # Remote query -- need to find all of them for the same service
                items = [item]
                for i in queue[:]:
                    if i.state == S_REMOTE and i.service is item.service: #@@ optimize which group is done first!
                        items.append(i)
                        queue.remove(i)
                nbs = query.remoteQuery(items)
                item.state = S_DONE  # do not put back on list
            elif state ==S_HEAVY_WAIT or state == S_LIGHT_WAIT:
                if item.quad[PRED] is query.store.universalVariableName or \
                   item.quad[PRED] is query.store.existentialVariableName:
                    ### We will never bind this variable in the first place
                    item.state = S_DONE
                    nbs = []
                else:
                    if diag.chatty_flag > 20 :
                        progress("@@@@ Warning: query can't find term which will work.")
                        progress( "   state is %s, queue length %i" % (state, len(queue)+1))
                        progress("@@ Current item: %s" % `item`)
                        progress(queueToString(queue))
                    return total  # Forget it
            else:
                raise RuntimeError, "Unknown state " + `state`
                
            if diag.chatty_flag > 90: progress("nbs=" + `nbs`)
            
#     query.py  Removed as I added this and wasn't sure whether it works with justReturn change below -tbl
#           # Optimization when sucess but no bindings
#           if (len(nbs) == 1 and nbs[0][0] == {} and nbs[0][1] is None and # if nbs == [({}, None)] and
#                   state == S_DONE):
#               if diag.chatty_flag>90: progress("LOOP to next, state="+`state`)
#               continue # Loop around and do the next one. optimization.

            for nb, reason in nbs:
                assert isinstance(nb,types.DictType), nb
                q2 = Queue([], queue)
                if query.justReturn:
                    ### What does the following do?
                    ### If we are doing a 1::1 match, record everything we have matched
                    if isinstance(reason, StoredStatement):
                        if reason not in q2.statements and \
                           reason[CONTEXT] is query.workingContext:
                            q2.statements.add(reason)
                        else:
                            continue
                if isinstance(reason, StoredStatement):
                    if True or reason[CONTEXT] is not query.workingContext:
                        for m in nb.values():
                            if isinstance(m, tuple):
                                m = m[0]
                            if m in reason[CONTEXT].existentials():
                                q2.bNodes.add(m)
                                if diag.chatty_flag > 80:
                                    ### These get picked up from log:includes
                                    ### {[:b :c] log:includes {?X :b :c} ...
                                    progress('Adding bNode %s, now %s' % (m, q2.bNodes))
                for i in queue:
                    newItem = i.clone()
                    q2.append(newItem)  #@@@@@@@@@@  If exactly 1 binding, loop (tail recurse)
                found = query.matchFormula(q2, variables.copy(), existentials.copy(),
                        bindings.copy(), nb, evidence = evidence + [reason])

                if diag.chatty_flag > 91: progress(
                    "Nested query returns %i (nb= %r)" % (found, nb))
                total = total + found
                if query.justOne and total:
                    return total

# NO - more to do return total # The called recursive calls above will have generated the output @@@@ <====XXXX
            if diag.chatty_flag > 80: progress("Item state %i, returning total %i" % (item.state, total))
            if (item.state == S_DONE):
                return total

            queue.append(item)
            # And loop back to take the next item

        if diag.chatty_flag>50: progress("QUERY MATCH COMPLETE with bindings: " + `bindings`)
        if query.justReturn:
            try:
                len(queue.statements)
                len(query.workingContext)
            except:
                print type(queue.statements)
                print type(query.workingContext)
                raise
            if len(queue.statements) != len(query.workingContext):
                return total
        newTotal = query.conclude(bindings,  evidence=evidence, extraBNodes = queue.bNodes, allBindings=bindings) + total  # No terms left .. success!
        if total:
            progress('newTotal=%s, total=%s' % (newTotal, total))
            raise RuntimeError('How did I get here?')
        return newTotal

    def remoteQuery(query, items):
        """Perform remote query as client on remote store
        Currently  this only goes to an SQL store, but should later use SPARQL etc
        in remote HTTP/SOAP call."""
        
        if diag.chatty_flag > 90:
            progress("    Remote service %s" % (items))
        serviceURI = items[0].service.uri
        if serviceURI.startswith("http:"):
            from sparql.sparqlClient import SparqlQuery
            return SparqlQuery(query, items, serviceURI)
        elif not serviceURI.startswith("mysql:"):
            raise ValueError("Unknown URI scheme for remote query service: %s" % serviceURI)
            
        import dbork.SqlDB
        from dbork.SqlDB import ResultSet, SqlDBAlgae, ShowStatement

        # SqlDB stores results in a ResultSet.
        rs = ResultSet()
        # QueryPiece qp stores query tree.
        qp = rs.buildQuerySetsFromCwm(items, query.variables, query._existentialVariables)
        # Extract access info from the first item.
        if diag.chatty_flag > 90:
            progress("    Remote service %s" %items[0].service.uri)
        (user, password, host, database) = re.match(
                "^mysql://(?:([^@:]+)(?::([^@]+))?)@?([^/]+)/([^/]+)/$",
                items[0].service.uri).groups()
        # Look for one of a set of pre-compiled rdb schemas.
        HostDB2SchemeMapping = { "mysql://root@localhost/w3c" : "AclSqlObjects" }
        if (HostDB2SchemeMapping.has_key(items[0].service.uri)):
            cachedSchema = HostDB2SchemeMapping.get(items[0].service.uri)
        else:
            cachedSchema = None
        # The SqlDBAlgae object knows how to compile SQL query from query tree qp.
        a = SqlDBAlgae(query.store.symbol(items[0].service.uri), cachedSchema,
            user, password, host, database, query.meta, query.store.pointsAt,
            query.store)
        # Execute the query.
        messages = []
        nextResults, nextStatements = a._processRow([], [], qp, rs, messages, {})
        # rs.results = nextResults # Store results as initial state for next use of rs.
        if diag.chatty_flag > 90: progress(string.join(messages, "\n"))
        if diag.chatty_flag > 90: progress("query matrix \"\"\""+
                        rs.toString({'dataFilter' : None})+"\"\"\" .\n")

        nbs = []
        reason = Because("Remote query") # could be messages[0] which is the query
        # Transform nextResults to format cwm expects.
        for resultsRow in nextResults:
            boundRow = {}
            for i in range(len(query.variables)):
                v = query.variables[i]
                index = rs.getVarIndex(v)
                interned = resultsRow[index]
                boundRow[v] = interned  # bindings
            nbs.append((boundRow, reason))

        if diag.chatty_flag > 10: progress("====> bindings from remote query:"+`nbs`)
        return nbs   # No bindings for testing

class BetterNone(object):
    __slots__ = []
    def __new__(cls):
        try:
            return cls.__val__
        except:
            cls.__val__ = object.__new__(cls)
        return cls.__val__
    def __hash__(self):
        raise TypeError
    def __str__(self):
        raise NotImplementedError
    def __eq__(self, other):
        raise NotImplementedError
    __neq__ = __eq__
    __lt__ = __gt__ = __leq__ = __geq__ = __eq__
BNone = BetterNone()

             
class QueryItem(StoredStatement):  # Why inherit? Could be useful, and is logical...
    """One line in a query being resolved.
    
    To a large extent, each line can keep track of its own state.
    When a remote query is done, query lines to the same server have to be coordinated again.
    """
    def __init__(self, query, quad):
        quad = lookupQuad(quad[CONTEXT]._redirections, quad)
        #print quad[CONTEXT]._redirections
        self.quad = quad
        self.query = query
        self.searchPattern = None # Will be list of variables
        self.store = query.store
        self.state = S_UNKNOWN  # Invalid
        self.short = INFINITY
        self.neededToRun = None   # see setup()
        self.myIndex = None     # will be list of satistfying statements
        self.service = None   # Remote database server for this predicate?
        self.builtIn = False
        return

    def clone(self):
        """Take a copy when for iterating on a query"""
        x = QueryItem(self.query, self.quad)
        x.state = self.state
        x.short = self.short
        x.neededToRun = []
        x.searchPattern = self.searchPattern[:]
        for p in ALL4:   # Deep copy!  Prevent crosstalk
            x.neededToRun.append(self.neededToRun[p].copy())  
        x.myIndex = self.myIndex
        try:
            x.interpretBuiltins = self.interpretBuiltins
        except AttributeError:
            pass
        return x



    def setup(self, allvars, unmatched, interpretBuiltins=[], mode=""):        
        """Check how many variables in this term,
        and how long it would take to search

        Returns, true normally or false if there is no way this query will work.
        Only called on virgin query item.
        The mode is a set of character flags about how we think."""
        con, pred, subj, obj = self.quad
        self.interpretBuiltins = interpretBuiltins
        self.service = None

        if diag.chatty_flag > 800: progress("setup:" + `allvars`)
        if "r" in mode:
            schema = None
            if "s" in mode:
                schema = pred.dereference(mode, self.query.workingContext)
                if schema != None:
                    if "a" in mode:
                        if diag.chatty_flag > 95:
                            progress("Axiom processing for %s" % (pred))
                        ns = pred.resource
                        rules = schema.any(subj=ns, pred=self.store.docRules)
                        rulefile = rulefile.dereference("m", self.query.workingContext)
                    self.service = schema.any(pred=self.store.definitiveService, subj=pred)
            if self.service == None and self.query.meta != None:
                self.service = self.query.meta.any(pred=self.store.definitiveService, subj=pred)
                if self.service == None:
                    uri = pred.uriref()
                    if uri[:4] == "mysql:":
                        j = uri.rfind("/")
                        if j>0: self.service = meta.newSymbol(uri[:j])
            if diag.chatty_flag > 90 and self.service:
                progress("We have a Remote service %s for %s." %(self.service, pred))
            if not self.service:
                authDoc = None
                if schema != None:
                    authDoc = schema.any(pred=self.store.definitiveDocument, subj=pred)
                if authDoc == None and self.query.meta != None:
                    authDoc = self.query.meta.any(pred=self.store.definitiveDocument, subj=pred)
                if authDoc != None:
                    if diag.chatty_flag > 90:
                        progress("We have a definitive document %s for %s." %(authDoc, pred))
                    authFormula = authDoc.dereference(mode, self.query.workingContext)
                    if authFormula != None:
                        self.quad = (authFormula, pred, subj, obj)
                        con = authFormula

        self.neededToRun = [ Set(), Set(), Set(), Set() ]  # for each part of speech
        self.searchPattern = [con, pred, subj, obj]  # What do we search for?
        hasUnboundCoumpundTerm = 0
        for p in PRED, SUBJ, OBJ :
            x = self.quad[p]
            if x in allvars:   # Variable
                self.neededToRun[p] = Set([x])
                self.searchPattern[p] = None   # can bind this
            elif isinstance(x, Formula) or isinstance(x, List): # expr  @@ Set  @@@@@@@@@@ Check and CompundTerm>???
                ur = x.occurringIn(allvars)
                self.neededToRun[p] = ur
                if ur != Set() or isinstance(x, Formula) or (isinstance(x, List) and hasFormula(x)):
                    hasUnboundCoumpundTerm = 1     # Can't search directly
                    self.searchPattern[p] = None   # can bind this if we recurse
                    
            if diag.chatty_flag > 98: progress("        %s needs to run: %s"%(`x`, `self.neededToRun[p]`))
                
        self.updateMyIndex(con)
        if isinstance(pred, RDFBuiltIn) or (
            interpretBuiltins and isinstance(pred, BuiltIn)):
            self.builtIn = True
        if isinstance(pred, RDFBuiltIn) or (
            interpretBuiltins and isinstance(pred, LightBuiltIn)):
            if self.canRun(): self.state = S_LIGHT_UNS_READY  # Can't do it here
            else: self.state = S_LIGHT_EARLY # Light built-in, can't run yet, not searched
        elif self.short == 0:  # Skip search if no possibilities!
            self.searchDone()
        elif hasUnboundCoumpundTerm:
            self.state = S_NEED_DEEP   # Put off till later than non-deep ones
        else:
            self.state = S_NOT_LIGHT   # Not a light built in, not searched.
        if diag.chatty_flag > 80: progress("setup:" + `self`)
        if self.state == S_DONE: return False
        return True

    def doIncludes(item, queue, existentials, variables, bindings):
        """Implement log:includes by adding all the statements on the LHS
        to the query items.  The plan is that this should"""
        con, pred, subj, obj = item.quad
        state = item.state
        
        nbs = []  # Failure
        if (isinstance(subj, Formula)
            and isinstance(obj, Formula)):
            oldsubj = subj
            subj = subj.renameVars()
            more_unmatched = buildPattern(subj, obj)
            more_variables = obj.variables().copy()
            _substitute({obj: subj}, more_unmatched)
            _substitute(bindings, more_unmatched)
            existentials = existentials | more_variables
            allvars = variables | existentials
            for quad in more_unmatched:
                newItem = QueryItem(item.query, quad)
                queue.append(newItem)
                if not newItem.setup(allvars, interpretBuiltins = 0,
                        unmatched=more_unmatched, mode=item.query.mode):
                    return []
            if diag.chatty_flag > 40:
                    progress("log:Includes: Adding %i new terms and %s as new existentials."%
                              (len(more_unmatched),
                               seqToString(more_variables)))
            rea = BecauseBuiltInWill(subj, con, oldsubj, pred, obj)
##          nbs = [({oldsubj: subj}, rea)]
            nbs = [(Env(), rea)]
        else:
#            if isinstance(subj, Formula): subj = subj.n3String()
#            if isinstance(obj, Formula): obj = obj.n3String()
            #raise RuntimeError("Cannot do {%s} log:includes {%s} " % (subj, obj))
            progress("""Warning: Type error ignored on builtin:
                log:includes only on formulae """+`item`)
                 #@@ was RuntimeError exception
            item.state = S_DONE
        return nbs


    def doSupports(item, queue, existentials, variables, bindings):
        """Real implementation of log:supports. Have fun.

        """
        con, pred, subj, obj = item.quad
        state = item.state
        store = con.store
        #statementObject = iter(con.statementsMatching(subj=subj, pred=pred, obj=obj)).next()

        nbs = []  # Failure
        if (isinstance(subj, Formula)
            and isinstance(obj, Formula)):
            assert subj.canonical != None
            F = store.any((store._experience, store.cufi, subj, None))  # Cached value?
            if F != None:
                if diag.chatty_flag > 10: progress("Bultin: " + `subj`+ " cached log:conclusion " + `F`)
            else:
                oldSubj = subj
                newSubj = subj.renameVars()

                F = subj.newFormula()
                newTopLevelFormula(F)
                if diag.tracking:
                    reason = Premise("Assumption of builtin", item)
                    #reason = BecauseMerge(F, subj)
    #           F.collector = reason
    #           proof.append(reason)
                else: reason = None
                if diag.chatty_flag > 10: progress("Bultin: " + `subj`+ " log:conclusion " + `F`)
                F.loadFormulaWithSubstitution(subj, why=reason, cannon=True) # leave open
                think(F)
                F4 = F.close()
                if F4 is not F:
                    raise RuntimeError
                    ### oh no! we've hit something old!
                    if F is subj:
                        ### absolute worst --- cannot be allowed!
                        F.add(F.newBlankNode(), F.newBlankNode(), F.newBlankNode())
                        F4 = F.close()
                F = F4
                assert subj.canonical != None
                
                store.storeQuad((store._experience, store.cufi, oldSubj, F),
                        why=BecauseOfExperience("conclusion"))  # Cache for later

            more_unmatched = buildPattern(F, obj)
            more_variables = obj.variables().copy()
            _substitute({obj: F}, more_unmatched)
            _substitute(bindings, more_unmatched)
            existentials = existentials | more_variables
            allvars = variables | existentials
            for quad in more_unmatched:
                newItem = QueryItem(item.query, quad)
                queue.append(newItem)
                if not newItem.setup(allvars, interpretBuiltins = 0,
                        unmatched=more_unmatched, mode=item.query.mode):
                    return []
            if diag.chatty_flag > 40:
                    progress("log:Includes: Adding %i new terms and %s as new existentials."%
                              (len(more_unmatched),
                               seqToString(more_variables)))
            rea = BecauseSupportsWill(con, subj, F, pred, obj)
##          nbs = [({oldsubj: subj}, rea)]
            nbs = [(Env(), rea)]
        else:
            if isinstance(subj, Formula): subj = subj.n3String()
            if isinstance(obj, Formula): obj = obj.n3String()
            #raise RuntimeError("Cannot do {%s} log:includes {%s} " % (subj, obj))
            progress("""Warning: Type error ignored on builtin:
                log:include only on formulae """+`item`)
                 #@@ was RuntimeError exception
            item.state = S_DONE
        return nbs



    def tryBuiltin(self, queue, bindings, evidence):                    
        """Check for  built-in functions to see whether it will resolve.
        Return codes:  0 - give up; 
                [] - success, no new bindings, (action has been called)
                [...] list of binding lists, each a pair of bindings and reason."""
        con, pred, subj, obj = self.quad
        proof = []  # place for built-in to hang a justification
        rea = None  # Reason for believing this item is true
        if "q" in self.query.mode:
            caughtErrors = (TypeError, ValueError, AttributeError, AssertionError, ArgumentNotLiteral, UnknownType)
        else:
            caughtErrors = ()
        try:
            if self.neededToRun[SUBJ] == Set():
                if self.neededToRun[OBJ] == Set():   # bound expression - we can evaluate it
                    try:
#                        if pred.eval(subj, obj,  queue, bindings.copy(), proof, self.query):
                        if pred.eval(subj, obj,  BNone, BNone, proof, BNone):
                            if diag.chatty_flag > 80: progress(
                                "Builtin buinary relation operator succeeds")
                            if diag.tracking:
                                rea = BecauseBuiltIn(con, subj, pred, obj)
                                return [(Env(), rea)]  # Involves extra recursion just to track reason
                            return [(Env(), None)]   # No new bindings but success in logical operator
                        else: return []   # We absoluteley know this won't match with this in it

                    except caughtErrors:
                        progress(
                        "Warning: Built-in %s %s %s failed because:\n   %s: %s"
                         % (`subj`, `pred`, `obj`, sys.exc_info()[0].__name__ , 
                             sys.exc_info()[1].__str__()  ))
                        if "h" in self.query.mode:
                            raise
                        return []
                    except BuiltinFeedError:
                        return []
                else: 
                    if isinstance(pred, Function):
                        if diag.chatty_flag > 97:
                            progress("Builtin function call %s(%s)"%(pred, subj))
                        try:
#                            result = pred.evalObj(subj, queue, bindings.copy(), proof, self.query)
                            result = pred.evalObj(subj, BNone, BNone, proof, BNone)
                        except caughtErrors:
                            errVal = (
                            "Warning: Built-in %s!%s failed because:\n   %s: %s"
                             % (`pred`, `subj`, sys.exc_info()[0].__name__ , 
                                 sys.exc_info()[1].__str__()  ))
                            progress(errVal)
                            if "h" in self.query.mode:
                                raise
                            if isinstance(pred, MultipleFunction):
                                result = [ErrorFlag(errVal)]
                            else:
                                result = ErrorFlag(errVal)
                        except BuiltinFeedError:
                            return []
                        if result != None:
                            self.state = S_DONE
                            rea=None
                            if isinstance(result, Formula):
                                if self.quad[OBJ] in self.query.variables:
                                    result.reopen()
                                    result = result.canonicalize(cannon=True)
                                    assert result.canonical is result
                            if diag.tracking:                                
                                if isinstance(result, Formula):
                                    result2 = result.renameVars()
                                    assert result.canonical is result, result.debugString()
                                else:
                                    result2 = result
                            if isinstance(pred, MultipleFunction):
                                return [(Env({obj:(x, None)}),
                                         diag.tracking and BecauseBuiltIn(con, subj, pred, x)
                                         ) for x in result]
                            else:
                                return [(nb1, diag.tracking and BecauseBuiltIn(con, subj, pred, result))
                                        for nb1, env3 in obj.unify(result, Env(), Env(), self.neededToRun[OBJ])]
##                                return [(Env({obj: (result, None)}),
##                                         diag.tracking and BecauseBuiltIn(con, subj, pred, result))]
                        else:
                            return []
            else:
                if (self.neededToRun[OBJ] == Set()):
                    if isinstance(pred, ReverseFunction):
                        if diag.chatty_flag > 97: progress("Builtin Rev function call %s(%s)"%(pred, obj))
                        try:
#                            result = pred.evalSubj(obj, queue, bindings.copy(), proof, self.query)
                            result = pred.evalSubj(obj, BNone, BNone, proof, BNone)
                        except caughtErrors:
                            errVal = (
                            "Warning: Built-in %s^%s failed because:\n   %s: %s"
                             % (`pred`, `obj`, sys.exc_info()[0].__name__ , 
                                 sys.exc_info()[1].__str__()  ))
                            progress(errVal)
                            if "h" in self.query.mode:
                                raise
                            if isinstance(pred, MultipleReverseFunction):
                                result = [ErrorFlag(errVal)]
                            else:
                                result = ErrorFlag(errVal)

                        except BuiltinFeedError:
                            return []
                        if result != None:
                            self.state = S_DONE
                            if isinstance(pred, MultipleReverseFunction):
                                return [(Env({subj:(x,None)}),
                                         diag.tracking and \
                                         BecauseBuiltIn(con, x, pred, obj)) \
                                        for x in result]
                            else:
                                return [(nb1, diag.tracking and BecauseBuiltIn(con, subj, pred, result))
                                        for nb1, env3 in subj.unify(result, Env(), Env(), self.neededToRun[SUBJ])]
##                                return [(Env({subj: (result,None)}),
##                                         diag.tracking and \
##                                         BecauseBuiltIn(con, result, pred, obj)
##                                         )]
                        else:
                            return []
                else:
                    if isinstance(pred, FiniteProperty):
                        result = pred.ennumerate()
                        if result != 0:
                            self.state = S_DONE
                            return result  # Generates its own list of (list of bindings, reason)s
                        else:
                            return []
            if diag.chatty_flag > 30:
                progress("Builtin could not give result"+`self`)
            return []   # no solution
            # @@@ remove dependency on 'heavy' above and remove heavy as param
        except (IOError, SyntaxError):
            raise BuiltInFailed(sys.exc_info(), self, pred ),None
        
    def tryDeepSearch(self, queue, oldBindings=Env()):
        """Search the store, unifying nested compound structures
        
        Returns lists of list of bindings, attempting if necessary to unify
        any nested compound terms. Added 20030810, not sure whether it is worth the
        execution time in practice. It could be left to a special magic built-in like "matches"
        or something if it is only occasionnaly used.
        
        Used in state S_NEED_DEEP
        """
        nbs = [] # Assume failure
        if self.short == INFINITY:
            if diag.chatty_flag > 36:
                progress( "Can't deep search for %s" % `self`)
        else:
            if diag.chatty_flag > 36:
                progress( "Searching (S=%i) %i for %s" %(self.state, self.short, `self`))
            try:
                for s in self.myIndex:
                    pass
            except:
                print self.myIndex, self
                raise
            for s in self.myIndex :  # for everything matching what we know,
                if self.query.justReturn and s in queue.statements:
                    continue
                env_queue = [(oldBindings, Env())]
                ### Why not just unify here?
                if diag.chatty_flag > 106: progress("...checking %r" % s)
                for p in PRED, SUBJ, OBJ:
                    if self.searchPattern[p] == None: # Need to check
                        eq = env_queue
                        env_queue = []
                        for nb, env2 in eq:
                            x = self.quad[p]
                            if self.neededToRun[p] == Set([x]):   # a term with no variables
                                if nb.canBind(x, s.quad[p]):
                                    nb = nb.bind(x, (s.quad[p], env2))
                                    env_queue.append((nb, env2))
                            else:  # Deep case   
                                if diag.chatty_flag > 70:
                                    progress( "Deep: Unify %s with %s vars=%s; ee=%s" %
                                    (x, s.quad[p], `self.query.variables`[4:-1],
                                    `self.query._existentialVariables`[4:-1]))
                                for nb1, env3 in x.unify(s.quad[p], nb, env2, self.neededToRun[p]):
                                    env_queue.append((nb1, env3))
##                                nbs1 = x.unify(s.quad[p], self.neededToRun[p] & self.query.variables,
##                                    self.neededToRun[p] & self.query._existentialVariables, {})  # Bindings have all been bound
                                
                for nb, _ in env_queue: 
                    nbs.append((nb, s))  # Add the new bindings into the set

        self.searchDone()  # State transitions
        return nbs   # End try deep search

    def searchDone(self):
        """Search has been done: figure out next state."""
        con, pred, subj, obj = self.quad
        if self.state == S_LIGHT_EARLY:   # Light, can't run yet.
            self.state = S_LIGHT_WAIT    # Search done, can't run
        elif self.state == S_LIGHT_UNS_READY: # Still have to run this light one
            return
        elif self.service:
            self.state = S_REMOTE    #  Search done, need to synchronize with other items
        elif not isinstance(pred, HeavyBuiltIn) or not self.interpretBuiltins:
            self.state = S_DONE  # Done with this one: Do new bindings & stop
        elif self.canRun():
            self.state = S_HEAVY_READY
        else:
            self.state = S_HEAVY_WAIT
        if diag.chatty_flag > 90:
            progress("...searchDone, now ",self)
        return
    
    def canRun(self):
        "Is this built-in ready to run?"

        if (self.neededToRun[SUBJ] == Set()):
            if (self.neededToRun[OBJ] == Set()): return 1
            else:
                pred = self.quad[PRED]
                return (isinstance(pred, Function)
                          or pred is self.store.includes
                          or pred is self.store.supports)  # Can use variables
        else:
            if (self.neededToRun[OBJ] == Set()):
                return isinstance(self.quad[PRED], ReverseFunction)


    def bindNew(self, newBindings):
        """Take into account new bindings from query processing to date

        The search may get easier, and builtins may become ready to run.
        Lists may either be matched against store by searching,
        and/or may turn out to be bound and therefore ready to run.
        
        Return:
            0           No way can this item ever complete.
            None        Binding done, see item.state 
        """
        con, pred, subj, obj = self.quad
        if diag.chatty_flag > 90:
            progress(" binding ", `self` + " with "+ `newBindings`)
        q=[con, pred, subj, obj]
        for p in ALL4:
            changed = 0
            for var, val in newBindings.items():
                if var in self.neededToRun[p]:
                    self.neededToRun[p].remove(var)
                    changed = 1
                if q[p] is var and self.searchPattern[p]==None:
                    self.searchPattern[p] = val # const now
                    changed = 1
                    self.neededToRun[p].clear() # Now it is definitely all bound
            if changed:
                q[p] = q[p].substitution(newBindings.asDict(), why=becauseSubexpression)   # possibly expensive
                if self.searchPattern[p] != None: self.searchPattern[p] = q[p]
                
        self.quad = q[0], q[1], q[2], q[3]  # yuk

        if self.state in [S_NOT_LIGHT, S_LIGHT_EARLY, S_NEED_DEEP, S_LIGHT_UNS_READY]: # Not searched yet
            hasUnboundCoumpundTerm = 0
            for p in PRED, SUBJ, OBJ :
                x = self.quad[p]
                if hasFormula(x): # expr  @@ Set  @@@@@@@@@@ Check and CompundTerm>???
#                    ur = x.freeVariables()
                    ur = x.occurringIn(self.neededToRun[p])
                    self.neededToRun[p] = ur
                    hasUnboundCoumpundTerm = 1     # Can't search directly
                    self.searchPattern[p] = None   # can bind this if we recurse

        
##            for p in PRED, SUBJ, OBJ:
##                x = self.quad[p]
##                if isinstance(x, Formula):
##                    if self.neededToRun[p]!= Set():
##                        self.short = INFINITY  # Forget it
##                        break
            else:
                self.updateMyIndex(con)

            if self.short == 0:
                self.searchDone()
        else:
            self.short = 7700+self.state  # Should not be used
            self.myIndex = None
        if isinstance(self.quad[PRED], BuiltIn):
            if self.canRun():
                if self.state == S_LIGHT_EARLY: self.state = S_LIGHT_UNS_READY
                elif self.state == S_LIGHT_WAIT: self.state = S_LIGHT_GO
                elif self.state == S_HEAVY_WAIT: self.state = S_HEAVY_READY
        if diag.chatty_flag > 90:
            progress("...bound becomes ", `self`)
        if self.state == S_DONE: return []
        return [({}, None)] # continue

    def updateMyIndex(self, formula):
        self.short, self.myIndex = formula.searchable(
                                            self.searchPattern[SUBJ],
                                           self.searchPattern[PRED],
                                           self.searchPattern[OBJ])
        if isinstance(self.quad[SUBJ], Formula) and self.short:
            self.short = self.short * 3 + len(self.quad[SUBJ]) * 100
        if isinstance(self.quad[OBJ], Formula) and self.short:
            self.short = self.short * 3 + len(self.quad[OBJ]) * 100
        for p in SUBJ, OBJ :
            if isinstance(self.quad[p], Formula) and not self.neededToRun[p]:
                newIndex = []
                for triple in self.myIndex:
                    if isinstance(triple[p], Formula) and len(triple[p]) == len(self.quad[p]):
                        newIndex.append(triple)
                self.myIndex = newIndex
        
##        self.myIndex = []
##        for triple in myIndex:
##            for loc in SUBJ, PRED, OBJ:
##                node = triple[loc]
##                if node in formula.variables() and self.searchPattern[loc] is not None:
##                    break
##            else:
##                self.myIndex.append(triple)

    def __repr__(self):
        """Diagnostic string only"""
        s = "%3i) short=%i, %s" % (
                self.state, self.short,
                quadToString(self.quad, self.neededToRun, self.searchPattern))
        if self.service: s += (", servce=%s " % self.service)
        return s


##############
# Compare two cyclic subsystem sto see which should be done first

def compareCyclics(self,other):
    """Note the set indirectly affected is the same for any
    member of a cyclic subsystem"""
    if other[0] in self[0].indirectlyAffects:
        return -1  # Do me earlier
    if other[0] in self[0].indirectlyAffectedBy:
        return 1
    return 0
        


#############  Substitution functions    


def _substitute(bindings, list):
    """Subsitustes IN-LINE into a list of quads"""
    for i in range(len(list)):
        q = list[i]
        list[i] = lookupQuad(bindings, q)
                            
def lookupQuad(bindings, q):
        "Return a subsituted quad"
        context, pred, subj, obj = q
        return (bindings.get(context, context),
                bindings.get(pred, pred),
                bindings.get(subj, subj),
                bindings.get(obj, obj))


def lookupQuadRecursive(bindings, q, why=None):
        context, pred, subj, obj = q
        if diag.chatty_flag > 99: progress("\tlookupQuadRecursive:", q)
        return (
            context.substitution(bindings),
            pred.substitution(bindings),
            subj.substitution(bindings),
            obj.substitution(bindings) )

#   DIAGNOSTIC STRING OUTPUT
#
def queueToString(queue):
    str = ""
    for item in queue:
        str = str  +  `item` + "\n"
    return str


def setToString(set):
    str = ""
    for q in set:
        str = str+ "        " + quadToString(q) + "\n"
    return str

def quadToString(q, neededToRun=[[],[],[],[]], pattern=[1,1,1,1]):
    qm=[" "," "," "," "]
    for p in ALL4:
        n = neededToRun[p]
        if n == []: qm[p]=""
        else: qm[p] = "(" + `n`[5:-2] + ")"  # Set([...]) ->  (...)
        if pattern[p]==None: qm[p]=qm[p]+"?"
    return "%s%s ::  %8s%s %8s%s %8s%s." %(`q[CONTEXT]`, qm[CONTEXT],
                                            `q[SUBJ]`,qm[SUBJ],
                                            `q[PRED]`,qm[PRED],
                                            `q[OBJ]`,qm[OBJ])

def seqToString(set):
#    return `set`
    set = list(set)
    str = ""
    for x in set[:-1]:
        str = str + `x` + ","
    for x in set[-1:]:
        str = str+ `x`
    return str

def bindingsToString(bindings):
#    return `bindings`

    str = ""
    for x, y in bindings.items():
        str = str + (" %s->%s " % ( `x`, `y`))
    return str

class BecauseBuiltInWill(object):
    def __init__(self, *args):
        self.args = args

class BecauseSupportsWill(object):
    def __init__(self, *args):
        self.args = args

class BuiltInFailed(Exception):
    def __init__(self, info, item, pred):
        progress("BuiltIn %s FAILED" % pred, `info`)
        self._item = item
        self._info = info
        self._pred = pred
        
    def __str__(self):
        reason = indentString(self._info[1].__str__())
#        return "reason=" + reason
        return ("Error during built-in operation\n%s\nbecause:\n%s" % (
            `self._item`,
#            `self._info`))
            `reason`))

def hasFormula(l):
    if not isinstance(l, (List, Formula)):
        return False
    if isinstance(l, Formula):
        return True
    for x in l:
        if hasFormula(x):
            return True
    return False

from term import AnonymousNode, CompoundTerm

def smarterSubstitution(f, bindings, source, why=None, exception=[]):
    bindings = bindings.asDict()
    if f in exception:
        return f
    if isinstance(f, Formula):
        f2 = f.newFormula()
        newBindings, _ = f2.loadFormulaWithSubstitution(f, bindings, why=Because("I said so #2", why))
        if f is not source:
            newExistentials = f2.occurringIn(source.existentials().intersection(Set(bindings.values())))
            for n in newExistentials:
                f2.declareExistential(n)
##            for k in bindings.values():
##                if k not in newExistentials and isinstance(k, AnonymousNode) and not isinstance(k,CompoundTerm) and f2.occurringIn(Set([k])) and len(f2) < 20:
##                    raise RuntimeError('How did I get here? newBindings=%s, bindings=%s, k=%s, f=%s, source=%s' % (newBindings, bindings, k, f2.debugString(), source.debugString()))
        rr = f2.close()
        return rr
    rr = f.substitution(bindings, why=Because("I said so #3", why))
    return rr

# ends
