""" Update for cwm architecture

The update module provides for the deletion as well as the addition of information to
a formula.  This module assumes the llyn.py store.  It connects intimiately with the
query module.

2004-03-17 written as an extension of query.py
"""


from set_importer import Set, sorted

import diag
from diag import chatty_flag, tracking, progress
from formula import Formula
from query import Query, Rule, seqToString, _substitute

    
def patch(workingContext, patchFormula):
    """A task of running a set of updates on a knowledge base
    
    This is simpler than an Inference task, in that a patch is only done
    once, patches cannot lead to new patches, etc.
    """
    if diag.chatty_flag >20:
        progress("New Update task, patches from %s applied to %s" %
                (patchFormula, workingContext))
    store = workingContext.store

    true = store.newFormula().close()  #   {}
    universals = Set()
    lhs_done = []
    agenda = {}
    for pred in store.insertion, store.deletion:
        for s in patchFormula.statementsMatching(pred=pred):
            dict = agenda.get(s.subject(), None)
            if dict == None:
                dict = {store.insertion: [], store.deletion: []}
                agenda[s.subject()] = dict
            dict[pred].append(s.object())
    for lhs, dict in sorted(agenda.items()):
        if diag.chatty_flag > 19: progress("Patches lhs= %s: %s" %(lhs, dict))
        if isinstance(lhs, Formula):
            if lhs.universals() != Set():
                raise RuntimeError("""Cannot query for universally quantified things.
                As of 2003/07/28 forAll x ...x cannot be on left hand side of rule.
                This/these were: %s\n""" % lhs.universals())
            
            addenda, minuenda = dict[store.insertion], dict[store.deletion]
            while addenda or minuenda:
                if addenda: conclusion = addenda.pop()
                else: conclusion = true
                if minuenda: retraction = minuenda.pop()
                else: retraction = true

                unmatched = lhs.statements[:]
                templateExistentials = lhs.existentials().copy()
                _substitute({lhs: workingContext}, unmatched)  # Change context column
            
                variablesMentioned = lhs.occurringIn(patchFormula.universals())
                variablesUsed = conclusion.occurringIn(variablesMentioned) | \
                                retraction.occurringIn(variablesMentioned)
                for x in sorted(variablesMentioned):
                    if x not in variablesUsed:
                        templateExistentials.add(x)
                if diag.chatty_flag >20:
                    progress("New Patch  =========== applied to %s" %(workingContext) )
                    for s in lhs.statements: progress("    ", `s`)
                    progress("+=>")
                    for s in conclusion.statements: progress("    ", `s`)
                    progress("-=>")
                    for s in retraction.statements: progress("    ", `s`)
                    progress("Universals declared in outer " + seqToString(patchFormula.universals()))
                    progress(" mentioned in template       " + seqToString(variablesMentioned))
                    progress(" also used in conclusion     " + seqToString(variablesUsed))
                    progress("Existentials in template     " + seqToString(templateExistentials))
    
                q = UpdateQuery(store, 
                        unmatched = unmatched,
                        template = lhs,
                        variables = patchFormula.universals(),
                        existentials =templateExistentials,
                        workingContext = workingContext,
                        conclusion = conclusion,
                        retraction = retraction,
                        rule = None)
                q.resolve()

        
class UpdateQuery(Query):
    "Subclass of query for doing patches onto the KB: adding and removing bits.  Aka KB Update"
    def __init__(self,
               store,
               unmatched,           # List of statements we are trying to match CORRUPTED
               template,                # formula
               variables,           # List of variables to match and return CORRUPTED
               existentials,        # List of variables to match to anything
                                    # Existentials or any kind of variable in subexpression
               workingContext,
               conclusion,      # Things to be added
               retraction,              # Things to be deleted
               rule):               # The rule statement

        Query.__init__(self, store, unmatched=unmatched, template=template, variables=variables,
                existentials= existentials,
                workingContext= workingContext,
                conclusion=conclusion, targetContext = workingContext, rule=rule)
        self.retraction = retraction

    def conclude(self, bindings, evidence = [], extraBNodes=Set(), allBindings=None):
        """When a match found in a query, add conclusions to target formula,
        and also remove retractions.

        Returns the number of statements added."""
        if diag.chatty_flag > 25: progress(
            "Insertions will now be made into %s. Bindings %s" % (self.workingContext, bindings))
        result = Query.conclude(self, bindings, evidence)
        
        # delete statements
        if diag.chatty_flag > 25: progress(
            "Insertions made, deletions will now be made. Bindings %s" % bindings)
        for st in self.retraction:
            s, p, o = st.spo()
            subj = self.doSubst(s, bindings)
            pred = self.doSubst(p, bindings)
            obj = self.doSubst(o, bindings)
            ss = self.workingContext.statementsMatching(
                    subj = subj,  pred = pred, obj = obj)
            if len(ss) != 1:
                progress("""Error: %i matches removing statement {%s %s %s} 
                    bound as {%s %s %s} from %s:\n%s\n""" %
                        (len(ss),s,p,o, subj.uriref(), pred.uriref(), obj.uriref(), self.workingContext, ss
                        ))
                progress(self.workingContext.debugString())
                raise RuntimeError(
                    """Error: %i matches removing statement {%s %s %s} 
                    bound as {%s %s %s} from %s""" %
                        (len(ss),s,p,o, `subj`, pred.uriref(), `obj`, self.workingContext))
            if diag.chatty_flag > 25: progress("Deleting %s" % ss[0])
            self.workingContext.removeStatement(ss[0])
        self.justOne = 1  # drop out of here when done
        return 1     # Success --  behave as a test and drop out

    def doSubst(self, x, bindings):
        if x.generated() and x not in bindings:
            raise ValueError("""Retractions cannot have bnodes in them.
            Use explict variables which also occur on the LHS.
            Found bnode: %s, bindings are: %s""" % (x, bindings))
        return bindings.get(x, x)


             
# ends
