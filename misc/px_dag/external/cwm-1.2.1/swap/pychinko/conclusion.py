#from  pychinko.interpreter import Interpreter
from terms import Formula, Pattern
from sets import Set
import sys
import os
import config

# TODO: fix bug: pychinko doesnot work for trivial cases, when there are no variables
"""{{<a> <b> <c>} log:implies {<test> a <SUCCESS> }.
  <a> <b> <c>."""

class Conclusion(object):
    """An implementation of CWM's log:conclusion"""
    def __init__(self, pattern):
        pass

    """run the rete on this the subj formula and return the result in obj"""
    def evaluate(self, subj, obj):
        #this is trickier:
        #create a rete, run it until no more new facts are found
        #return back the formula
        print "here!"

        
        rules = subj.rules
        #there may be facts in the rules file
        facts = subj.facts
                
        interp = Interpreter(rules)

        #print "patterns:"
        # for i in  subj.patterns:
        #    print i
        for i in rules:
            print "rule:", i
        #print "Ground fact(s): ", len(facts)
        #print facts
                
        
        interp.addFacts(Set(facts), initialSet=True)
        
        
        interp.run()

        print "interpreter sucessfully ran"
        print len(interp.inferredFacts)

        #create a new formula with the new facts + the stuff in subj and return it
        f = Formula()
        # copy all of the patterns
        f.patterns.extend(subj.patterns)

        #add the new ones
        for i in interp.inferredFacts:
            f.patterns.append(Pattern(i.s, i.p, i.o))
                        
        f.rules = f.getRules()
        f.facts = f.getFacts()

        return f        