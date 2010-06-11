#
# The Rete network
#

from nodes import AlphaIndex, BetaIndex, AlphaNode, BetaNode, RuleNode, AlphaStore, BetaStore, QueryRuleNode, py_lists
from terms import Pattern, URI
from helpers import removedups, sortVars, keysToList, listToPython
from builtins import Builtin, CWMBuiltins, builtinp, getPatternType

###
### We make sure to build Rete network such that more "selective"
### JOINs appear first, e.g.:
###
###   LHS: (?x p ?y) (?z p ?a) (?a p ?d)
###
### In this case, it would clearly be more efficient to build the network such that
### the first and second ANodes are (?z p ?a) (?a p ?d) since they are likely to be more
### selective than (?x p ?y)
### 
class Rete(object):
    def __init__(self):
        self.alphaNodeStore = AlphaStore()
        self.betaNodeStore = BetaStore()
        self.betaIndex = BetaIndex()
        self.alphaIndex = AlphaIndex()
        
        
    def printNetwork(self):
        for bnode in self.betaNodeStore:
            print bnode
    
class RuleCompiler(object):
    def __init__(self):
        self.rete = Rete()


    def addRule(self, rule):        
            alphaNodesOfRule = AlphaStore()
            for pattern in rule.lhs:
                anodePattern = Pattern(pattern.s, pattern.p, pattern.o)
                anode = self.makeAlphaNode(anodePattern)                
                alphaNodesOfRule.addNode(anode)
        
            alphaNodesOfRule.sort()
            self.rete.alphaNodeStore.sort()
            alphaNodesOfRule = removedups(alphaNodesOfRule)
            
            l = len(alphaNodesOfRule)
            if l == 0:
                # probably malformed input
                raise Exception
            elif l == 1:
                # If the rule has one pattern, we create two identical anodes
                # for it so that we can have a BetaNode with a left and right input
                beta1 = self.makeBetaNode(alphaNodesOfRule[0], alphaNodesOfRule[0],
                                          futureJoins=False)
                alphaNodesOfRule[0].betaNodes = [beta1]
                beta1.rule = self.makeRuleNode(rule)
                beta1.rule.betaNode = beta1                                
            elif l == 2:
                # here we build a one beta with the only two alphas as its inputs
                beta1 = self.makeBetaNode(alphaNodesOfRule[0], alphaNodesOfRule[1],
                                          futureJoins=False)
                # connect our AlphaNodes to the BetaNode
                alphaNodesOfRule[0].betaNodes = [beta1]
                alphaNodesOfRule[1].betaNodes = [beta1]
                beta1.rule = self.makeRuleNode(rule)
                beta1.rule.betaNode = beta1
                self.rete.betaNodeStore.addNode(beta1)
            else:
                beta1 = self.makeBetaNode(alphaNodesOfRule[0], alphaNodesOfRule[1],
                                          futureJoins=True)
                alphaNodesOfRule[0].betaNodes = [beta1]
                alphaNodesOfRule[1].betaNodes = [beta1]
                self.rete.betaNodeStore.addNode(beta1)
                # we've consumed the first two alpha nodes
                alphaNodeList = alphaNodesOfRule[2:]
                self.makeBetaNetwork(rule, beta1, alphaNodeList)
                
            return self.rete, alphaNodesOfRule

    
    def compile(self, rules):        
        for rule in rules:
            print "%s of %s" % (rules.index(rule), len(rules))
            alphaNodesOfRule = AlphaStore()
            for pattern in rule.lhs:
                anodePattern = Pattern(pattern.s, pattern.p, pattern.o)
                anode = self.makeAlphaNode(anodePattern)                
                alphaNodesOfRule.addNode(anode)
        
            alphaNodesOfRule.sort()
            self.rete.alphaNodeStore.sort()
            alphaNodesOfRule = removedups(alphaNodesOfRule)
            
            l = len(alphaNodesOfRule)
            if l == 0:
                # probably malformed input
                raise Exception
            elif l == 1:
                # If the rule has one pattern, we create two identical anodes
                # for it so that we can have a BetaNode with a left and right input
                beta1 = self.makeBetaNode(alphaNodesOfRule[0], alphaNodesOfRule[0],
                                          futureJoins=False)
                alphaNodesOfRule[0].betaNodes = [beta1]
                beta1.rule = self.makeRuleNode(rule)
                beta1.rule.betaNode = beta1                                
            elif l == 2:
                # here we build a one beta with the only two alphas as its inputs
                beta1 = self.makeBetaNode(alphaNodesOfRule[0], alphaNodesOfRule[1],
                                          futureJoins=False)
                # connect our AlphaNodes to the BetaNode
                alphaNodesOfRule[0].betaNodes = [beta1]
                alphaNodesOfRule[1].betaNodes = [beta1]
                beta1.rule = self.makeRuleNode(rule)
                beta1.rule.betaNode = beta1
                self.rete.betaNodeStore.addNode(beta1)
            else:
                beta1 = self.makeBetaNode(alphaNodesOfRule[0], alphaNodesOfRule[1],
                                          futureJoins=True)
                alphaNodesOfRule[0].betaNodes = [beta1]
                alphaNodesOfRule[1].betaNodes = [beta1]
                self.rete.betaNodeStore.addNode(beta1)
                # we've consumed the first two alpha nodes
                alphaNodeList = alphaNodesOfRule[2:]
                self.makeBetaNetwork(rule, beta1, alphaNodeList)
        return self.rete

    def makeBetaNetwork(self, rule, betaNode, alphaNodeList):
        """I have more than 2 alpha nodes and so I make a network"""
        length = len(alphaNodeList)
        if length == 0:
            betaNode.rule = self.makeRuleNode(rule)
            betaNode.rule.betaNode = betaNode
        else:
            alpha = alphaNodeList[0]
            betaChild = self.makeBetaNode(betaNode, alpha)
            # connect our newly created BetaNode to its parent BetaNode,
            # and connect the parent to its child
            betaChild.parents = [betaNode]
            betaNode.children = [betaChild]
            sharedJoinVars = self.getSharedVars(betaNode, alpha)
            sortVars(sharedJoinVars)
            if not builtinp(alpha):
                # adjust our beta node shared variables
                betaNode.svars = sharedJoinVars
                # Our betanode has children, and so set up our
                # pattern in an order that is conducive to the children JOINs
                betaNode.pattern = removedups(sharedJoinVars + betaNode.pattern)
            # connect our AlphaNode to its relevant BetaNode
            alpha.betaNodes = [betaChild]
            self.rete.betaNodeStore.addNode(betaNode)
            self.rete.betaNodeStore.addNode(betaChild)
            alphaNodeList = alphaNodeList[1:]
            return self.makeBetaNetwork(rule, betaChild, alphaNodeList)

    def getSharedVars(self, node1, node2):
        """I return a list of shared variables between two nodes"""
        lvars = list()
        rvars = list()
        # check for builtins - do not calculate the shared variables
        # between a builtin and a regular Beta Node, since the index order
        # does not change
        if isinstance(node1, BetaNode):
            lvars = node1.pattern
        else:
            lvars = node1.vars
        if isinstance(node2, BetaNode):
            rvars = node2.pattern
        else:
            rvars = node2.vars       
        result = removedups([item for item in lvars \
                             if item in rvars])        
        return result

    def makeBetaNode(self, node1, node2, futureJoins=True):
        sharedVars = self.getSharedVars(node1, node2)        
        # if our left input is an alpha node, then it has the same shared
        # variables as our right input
        if isinstance(node1, AlphaNode):
            node1.svars = sharedVars
            sortVars(node1.svars)
        node2.svars = sharedVars
        sortVars(node2.svars)
        # make new BetaNode
        b = BetaNode(node1, node2)
        # store the shared variables in our new BetaNode
        # the shared vars here will be reset if the beta node has beta children
        b.svars = sharedVars
        # just add the beta node to the index (which is a flat list), no indexing for now
        return b
    
    def makeAlphaNode(self, pattern):
        # check if it's a builtin, and if it is, instantiate it
        biType = getPatternType(URI(pattern.p))
        if biType:
            a = biType(pattern)            
        else:
            a = AlphaNode(pattern)
            
        self.rete.alphaIndex.add(a)
        self.rete.alphaNodeStore.addNode(a)        
        return a

    def makeRuleNode(self, rule):
        """I make a new RuleNode (which will be linked to the last BetaNode)"""
        # if the right hand side is empty, treat it as a query rule node
        if not rule.rhs:
            return QueryRuleNode() # A dummy
        else:
            return RuleNode(rule)
