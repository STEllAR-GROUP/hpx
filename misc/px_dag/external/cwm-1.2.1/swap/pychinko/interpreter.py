import sys

import rete
from terms import Variable, Pattern, Rule, Fact, URI, makeToken, Exivar, Univar
from nodes import AlphaIndex, BetaNode, QueryRuleNode, py_lists
from sets import Set
from helpers import removedups, TYPE, listToPython, aNodeListToPython, factListToPython, FIRST, REST, LIST, INCLUDES, convertBNodeToFact, isListVarNode
#from pychinko import N3Loader
import time

class Interpreter:
    def __init__(self, rules=None):
        self.compiler = rete.RuleCompiler()
        self.rules = rules        
        self.rete = self.compiler.compile(rules)
        self.totalFacts = Set()
        self.joinedBetaNodes = Set()
        self.inferredFacts = Set()
        self.fta = Set()  #FactsToAdd
        self.rulesThatFired = Set()
        self.initialFacts = Set()
                
                      
    def addRule(self, rule):
        #self.rules.add(rule)                   
        self.rete, alphaNodesOfRule = self.compiler.addRule(rule)          
        #add facts only to the alphanodes of this rule
        self.addFactsForRule(self.totalFacts,  alphaNodesOfRule)        
                
    #this function propagates the totalFacts to the alphanodes of the rule    
    def addFactsForRule(self, facts, alphaNodesRule):
        status = False      
        t = time.time()
        listFacts = Set()
        if not alphaNodesRule:
            print "No alpha nodes in rule!"
            return False        

        for f in facts:
            alphaMatches = self.rete.alphaIndex.match(f)            
            for anode in alphaMatches:
                if anode in alphaNodesRule:                
                    if anode.add(f):                    
                        status = True        
    
        print "add facts for new rule time:", time.time() - t
        return status #bool(alphaMatches)  #true if any alpha nodes matched

              
    #clean up this function, make it more flexible so when we add only 1 new fact we do not have to rerun the whole thing    
    def addFacts(self, facts, initialSet=False): #facts = Set
        """Adds a set of facts to the AlphaNodes."""
        if initialSet:
            self.initialFacts = Set(facts)
        else:
            pass
            #self.initialFacts  = self.totalFacts #not an initial set, then initialFacts would be the old total facts
                
        if not self.rete.alphaIndex:
            print "No alpha nodes in store!"
            return False
        #add facts for every list-related pattern with bnodes
        for alphaNode in self.rete.alphaNodeStore:            
            #if the alphanode includes bnodes (list related) then add a matching fact            
            if (alphaNode.pattern.p in [FIRST, REST, INCLUDES]) or (alphaNode.pattern.p==TYPE and alphaNode.pattern.o==LIST):                
                #add a fact                
                subj = convertBNodeToFact(alphaNode.pattern.s)
                obj = convertBNodeToFact(alphaNode.pattern.o)                
                if not isinstance(subj, Variable) and not isinstance(obj, Variable):                    
                    #add it to the set of initial facts or the alphanode?
                    #add the fact directly to the alphanode for now, dont have to do the matching
                    #self.initialFacts.add(Fact(subj, alphaNode.pattern.p, obj))                    
                    #alphaNode.add(Fact(subj, alphaNode.pattern.p, obj))  
                    #print "adding fact"
                    #print subj, alphaNode.pattern.p, obj
                    facts.add(Fact(subj, alphaNode.pattern.p, obj))     
                                   

        #following code is converting and adding python lists in dictionary py_lists 
        #for the rules triples that are list related       
        for alphaNode in self.rete.alphaNodeStore:
            #print alphaNode
            if (alphaNode.pattern.p == FIRST and not isinstance(alphaNode.pattern.s, Univar) ):
                #print "bnode:", convertBNodeToFact(alphaNode.pattern.s)
                py_list = aNodeListToPython(convertBNodeToFact(alphaNode.pattern.s), self.rete.alphaNodeStore.nodes, [])
                #py_list = listToPython(convertBNodeToFact(alphaNode.pattern.s), self.rete.alphaNodeStore.nodes, self.initialFacts.union(facts),[])
                
                py_lists[convertBNodeToFact(alphaNode.pattern.s)] = py_list
                #print "py_lists:", py_lists
        
        status = False
        
        #this code does the same for the facts
        for f in self.initialFacts:
            if f.p==FIRST:                
                py_list = factListToPython(convertBNodeToFact(f.s), self.initialFacts.union(facts),[])
                py_lists[convertBNodeToFact(f.s)] = py_list
                #print "py_lists:", py_lists
                
            self.totalFacts.add(f)
            #add only to relevant alpha nodes
            alphaMatches = self.rete.alphaIndex.match(f)            
            for anode in alphaMatches:                
                if anode.add(f):                    
                    status = True
        
        for f in facts.difference(self.initialFacts):            
            #if this fact is of type list, add it to the dict
            if f.p==TYPE and f.o==LIST:                
                py_list = factListToPython(convertBNodeToFact(f.s), self.initialFacts.union(facts),[])
                py_lists[convertBNodeToFact(f.s)] = py_list
                #print "py_lists:", py_lists
                
            self.totalFacts.add(f)
            #add only to relevant alpha nodes
            alphaMatches = self.rete.alphaIndex.match(f)
            
            for anode in alphaMatches:
                #print "new fact added:", f
                #print "_to alpha node added:", anode.pattern
                #do not add the fact if the bnode does not match
                #this is to take care of: _:xsdiuh rdf:first ?y or _:sdfds rdf:rest _:dsfdsf
                if (anode.pattern.p in [REST, FIRST]) or (anode.pattern.p==TYPE and anode.pattern.o==LIST):
                    if isinstance(anode.pattern.s, Exivar) and convertBNodeToFact(anode.pattern.s)==f.s:
                        if anode.add(f):
                            status = True
                else:
                    if anode.add(f):
                        status = True
                        
                #if isinstance(anode.pattern.s, Exivar) and convertBNodeToFact(anode.pattern.s)==f.s:
#                if anode.add(f):
 #                   status = True
                #elif not isinstance(anode.pattern.s, Exivar):
                #    if anode.add(f):                    
                 #       status = True
        
        return status #bool(alphaMatches)  #true if any alpha nodes matched

   
    def run(self, alphaMatches = None):        
        """I walk the Rete network and attempt to make the necessary JOINs."""            
        t = time.time()
        for alphaNode in self.rete.alphaNodeStore:     
            for betaNode in alphaNode.betaNodes:                
                if betaNode in self.joinedBetaNodes:                    
                    continue
                self.fta = Set() #resets the list of inferred facts from JOINs
                self.processBetaNode(betaNode)  #returns results via self.fta
            if self.fta:
                #facts that are in newFacts but not in inferred facts = "unique facts"
                # print "inferred facts:", self.fta
                self.fta = self.fta.difference(self.initialFacts)
                newInferredFacts = self.fta.difference(self.inferredFacts)
                if newInferredFacts:
                    self.inferredFacts.union_update(newInferredFacts)  
                    #reapply the rules to the new facts. If unique facts is empty or
                    #the new facts are a subset of the initial facts, we terminate.
                    if self.addFacts(newInferredFacts):
                        self.joinedBetaNodes = Set()
                        self.run()

        # take the inferred facts, create an rdflib.graph and serialize it in rdf/xml
        #this is a rules in PIT extension
        #N3loader.convertBackToGraph(self.inferredFacts)
                        
        
        print "interpreter run() time:", time.time() - t                                
    

    def processBetaNode(self, betaNode):        
        """I process a beta node"""
        #t = time.time()
        inferences = betaNode.join()
        
        if inferences:
            self.joinedBetaNodes.add(betaNode)
            if betaNode.rule:
                #self.rulesThatFired.add(betaNode.rule)
                #######this test will be moved into `matchingFacts'
                for rhsPattern in betaNode.rule.rhs:
                    results = betaNode.matchingFacts(rhsPattern, inferences)
                    self.fta.union_update(Set(results))
            else:
                for child in betaNode.children:
                    #process children of BetaNode..
                    self.processBetaNode(child)
        #print "process betanode time:", time.time() - t        
