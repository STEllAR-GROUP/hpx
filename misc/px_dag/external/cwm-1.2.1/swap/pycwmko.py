#!/usr/bin/env python
"""

"""

from pychinko import terms, interpreter
#from pychinko import N3Loader
from pychinko.helpers import removedups, convertBNodeToFact
from swap import term, formula
from swap.set_importer import Set
import time
#from rdflib import BNode, Store
# from rdflib.constants import TYPE, FIRST, REST, LIST, NIL, OWLNS
LOG_IMPLIES = 'http://www.w3.org/2000/10/swap/log#'

try:
    reversed
except NameError:
    def reversed(l):
        ll = [a for a in l]
        b = len(l)
        while b > 0:
            b -= 1
            yield ll[b]


from pychinko import nodes, rete

class fullSet(object):
    def __contains__(self, other):
        return True

from sys import stderr

class directPychinkoQuery(object):
    def __init__(self, workingContext, rulesFormula=None, target=None):
            
        self.extra = []
        self.store = workingContext.store
        self.workingContext = workingContext
        if rulesFormula is None:
            rulesFormula = workingContext
        if target is None:
            target = workingContext
        t = time.time()
        self.rules = self.buildRules(rulesFormula)
        self.interp = interpreter.Interpreter(self.rules)
        #print "rules"
        #print self.rules
                
        self.facts = self.buildFacts(rulesFormula)
        print "converting and adding time:", time.time() - t
        t = time.time()
                
                       
            
                
#        self.interp.addFacts(Set(self.facts), initialSet=True)
        #print self.rules
        #print "add facts time:", time.time() - t
        t = time.time()
        self.interp.run()
        print "interp.run() time:", time.time() - t

        print len(self.interp.inferredFacts), ' inferred fact(s)'
        #print "size of inferred facts:", len(self.interp.inferredFacts)
#        print self.interp.inferredFacts
        # add the inferred facts back to cwm store
        t = time.time()
        for i in self.interp.inferredFacts:
#                convertFromPystore();
#                if isinstance(i.o, str):
#                        print type(i.o)
#                elif isinstance(i.o, unicode):
#                        print('unicode')
                # convert them to term.Symbols 
                # cannot convert to term.Symbol if it's a literal
#                print i.s, i.p, i.o
#                print self.convFromRete(i.s),  self.convFromRete(i.p), self.convFromRete(i.o)

                newTriple =  self.convFromRete(i.s),  self.convFromRete(i.p), self.convFromRete(i.o)
                self.workingContext.add(*newTriple)
#                if not  self.workingContext.contains(newTriple):
#                        self.workingContext.add(*newTriple)
#                else:
#                        print "contains!"
        print "add facts time to cwm:", time.time() - t

        """
        print "facts"
        print self.facts
        self.workingContext = workingContext
        self.target = target
        if workingContext is target:
            self.loop = True
        else:
            self.loop = False"""
        

    def convFromRete(self, t):
            if not t:
                    return None
#            print "cnv:", t, type(t)            
            if isinstance(t,unicode):                    
                    return self.workingContext.newSymbol(t)
            elif isinstance(t,str):
                    return self.workingContext.newLiteral(t)
            return term.Symbol(t, self.store)
            
    def convType(self, t, F, K=None):  
#        print "type:",t, type(t)
        """print "t:", t
        print type(t)
        print "f unis:", F.universals()
        if (K): print "k exis:", K.existentials()"""
        
        if isinstance(t, term.NonEmptyList):
            #convert the name of the list to an exivar and return it            
            #self.convertListToRDF(t, listId, self.extra)
            #return terms.Exivar('_:' + str(t))
            return '_:' + str(t)
            #raise RuntimeError            
        if t in F.universals():
            return terms.Variable(t)
        if K is not None and t in K.existentials():
#            print "returning existential:", t
            return terms.Exivar(t)
        if isinstance(t, term.Symbol):
                return terms.URIRef(t.uri)
        if isinstance(t, term.BuiltIn):
                return t.uriref()
        if isinstance(t, term.Fragment):
#                print "uriref:",terms.URIRef(t.uriref())
#                print type(t.uriref())
                return terms.URIRef(t.uriref())
#        print type(t)
#        print "returning URI",t              
        return str(t)
                    
    """def processBetaNode(self, betaNode):        
        retVal = False
        inferences = betaNode.join()
        self.joinedBetaNodes.add(betaNode)
        if inferences:
            if betaNode.rule:
                #self.rulesThatFired.add(betaNode.rule)
                #######this test will be moved into `matchingFacts'
                for rhsPattern in betaNode.rule.rhs:
                    results = betaNode.matchingFacts(rhsPattern, inferences)
                    ### @@@ here we need to add to the workingcontext
                    for triple in results:
                        addedResult = self.workingContext.add(*triple.t)
                        if addedResult:
                            retVal = True
                            self.newStatements.add(
                                self.workingContext.statementsMatching(
                                    subj=triple.s, pred=triple.p, obj=triple.o)[0])
#                        retVal = retVal or addedResult
            else:
                for child in betaNode.children:
                    #process children of BetaNode..
                    betaNodeProcessed = self.processBetaNode(child)
                    retVal = retVal or betaNodeProcessed
        return retVal"""
                 
    def _listsWithinLists(self, L, lists):
        if L not in lists:
            lists.append(L)
        for i in L:
            if isinstance(i, term.NonEmptyList):
                self._listsWithinLists(i, lists)

    def dumpLists(self, context):
        "Dump lists out as first and rest. Not used in pretty."        
        listList = {}
        result = []
        #context = self.workingContext
        #sink = self.sink
        lists = []
        for s in context.statements:
            #print "s:", s
            for x in s.predicate(), s.subject(), s.object():
                if isinstance(x, term.NonEmptyList):
                    self._listsWithinLists(x, lists)
                    
        for l in lists:           
            list = l
            while not isinstance(list, term.EmptyList):
                if list not in listList:
                    #print list, " rdf:type rdf:list"
                    #self._outputStatement(sink, (context, self.store.forSome, context, list))
                    listList[list] = 1
                list = list.rest
                
        listList = {}
        for l in lists:
            list = l            
            while (not isinstance(list, term.EmptyList)) and list not in listList:    
                result.append(terms.Pattern(terms.Exivar("_:" + str(list)), "http://www.w3.org/1999/02/22-rdf-syntax-ns#first", self.convType(list.first, self.workingContext, context)))                
                if isinstance(list.rest, term.EmptyList):
                        #print "_:", list, " rdf:rest rdf:nil"
                        result.append(terms.Pattern(terms.Exivar("_:" + str(list)), "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest", "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"))                        
                else:    
                        result.append(terms.Pattern(terms.Exivar("_:" + str(list)), "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest", self.convType(list.rest, self.workingContext, context)))    
                        #print list, " rdf:rest ", list.rest
                #self._outputStatement(sink, (context, self.store.first, list, list.first))
                #self._outputStatement(sink, (context, self.store.rest,  list, list.rest))
                listList[list] = 1
                list = list.rest
                
        return result
                 
                    
    def buildRules(self, indexedFormula):
        rules = []
        for rule in indexedFormula.statementsMatching(pred=indexedFormula.store.implies):
            subj, predi, obj = rule.spo()
            
            if not isinstance(subj, formula.Formula) or \
               not isinstance(obj, formula.Formula):
                continue
            head = []
            tail = []
            for fr, to in (subj, tail), (obj, head):                
                self.extra = self.dumpLists(fr) #use extra for the list-related triples                   
                for quad in fr: 
                    #if not isinstance(quad.subject(), term.NonEmptyList):                                                           
                    s, p, o = [self.convType(x, indexedFormula, fr)
                               for x in quad.spo()] #to get variables.
                               #Not good enough for Lists   
#                    print "spo:", s,p,o
                    for f in (self.extra + [(s,p,o)]):
                        to.append(terms.Pattern(*f))
            rules.append(terms.Rule(tail, head, (subj, predi, obj) ))
            
        return rules

    def buildFacts(self, indexedFormula):
        facts = []
        for f in self.dumpLists(indexedFormula):                
                facts.append(terms.Fact(convertBNodeToFact(f.s),f.p, convertBNodeToFact(f.o)))
           
        
#        for alphaNode in self.interp.rete.alphaNodeStore:
#                print alphaNode
#                i = alphaNode.pattern.noneBasedPattern()
#                pattern =  self.convFromRete(i[0]),  self.convFromRete(i[1]), self.convFromRete(i[2])
#                print "pattern:", pattern
#                for quad in indexedFormula.statementsMatching(
#                    subj=pattern[0],
#                    pred=pattern[1],
#                    obj =pattern[2]):                    
##                    print "quad:", quad
#                    if  isinstance(subj, formula.Formula) or isinstance(obj, formula.Formula):
#                            print "The RETE engine cannot process nested formulas currently"
#                            continue
#                    
#                    s, p, o = [self.convType(x, indexedFormula, None) for x in quad.spo()]
#                    
#                    alphaNode.add(terms.Fact(s,p,o))
     
        for fact in indexedFormula.statements:
            subj, predi, obj = fact.spo()
            # ignore formulas for now
            
            if  isinstance(subj, formula.Formula) or \
                isinstance(obj, formula.Formula):
                
                 print "The RETE cannot process nested formulas at the time - use it for ntriples only"
#                raise NotImplementedError
                
                 continue
            # only get top level facts            
            head = []
            tail = []            
            
            
            s, p, o = [self.convType(x, indexedFormula, None)
                       for x in fact.spo()] #to get variables.
                       #Not good enough for Lists, but they're taken care of earlier
                      
            facts.append(terms.Fact(s, p, o))

        self.interp.addFacts(Set(facts), initialSet=True)             
        return facts


    def add(self, triple):
        t = triple.t
        status = False
        if self.workingContext.add(*t):
            alphaMatches = self.rete.alphaIndex.match(f)
            for anode in alphaMatches:
                if anode.add(f):
                    status = True
        return Status

"""      

 def __call__(self):
        #convert it to a set of facts (simply take all triples in a formula and add them as facts)
        #as first cut
        rules = self.rules
        indexedFormula = self.workingContext
        self.newStatements = fullSet()
        self.rete = rete.RuleCompiler().compile(rules)
        newStuff = True
        first = True

        while newStuff and (first or self.loop):
            #print >> stderr, "starting loop"
            first = False
            newStuff = False
            needToRun = False
            for alphaNode in self.rete.alphaNodeStore:
                pattern = alphaNode.pattern.noneBasedPattern()
                for quad in indexedFormula.statementsMatching(
                    subj=pattern[0],
                    pred=pattern[1],
                    obj =pattern[2]):
                    self.extra = []
                    if quad in self.newStatements:
                        s, p, o = [self.convType(x, indexedFormula)
                                   for x in quad.spo()]
                        for f in (self.extra + [(s,p,o)]):
                            if alphaNode.add(terms.Fact(*f)):
                                needToRun = True
                                
            self.newStatements = Set()                
            self.joinedBetaNodes = Set()
            if needToRun:
                for alphaNode in self.rete.alphaNodeStore:
                    for betaNode in alphaNode.betaNodes:
                        if betaNode in self.joinedBetaNodes:
                            continue
                        newNewStuff = self.processBetaNode(betaNode)
                        newStuff = newStuff or newNewStuff
        
        print self.newStatements
#        self.rete.printNetwork()
          
class ToPyStore(object):

    def __init__(self, pyStore):
        self.pyStore = pyStore
        self.lists = {}
        self.typeConvertors = [ 
            (formula.Formula , self.formula),  
            (formula.StoredStatement, self.triple),
            (term.LabelledNode, self.URI), 
            (term.Fragment, self.URI), 
            (term.AnonymousNode, self.BNode),
            (term.Literal, self.literal),
            (term.List, self.list),
            (term.N3Set, self.set)]

    def lookup(self, node):
        for theType, function in self.typeConvertors:
            if isinstance(node, theType):
                return function(node)
        raise RuntimeError(`node` + '  ' + `node.__class__`)

    def formula(self, node):
        subFormulaRef = self.pyStore.create_clause()
        subFormula = self.pyStore.get_clause(subFormulaRef)
        subConvertor = self.__class__(subFormula)
        subConvertor.statements(node)
        return subFormulaRef

    def URI(self, node):
        return terms.URI(node.uriref())

    def BNode(self, node):
        return BNode.BNode(node.uriref())

    def literal(self, node):
        string = node.string
        dt = node.datatype
        if not dt:
            dt = ''
        lang = node.lang
        if not lang:
            lang = ''
        return terms.Literal(string, lang, dt)

    def list(self, node):
        if node in self.lists:
            return self.lists[node]
        newList = [].__class__
        next = NIL
        for item in reversed(newList(node)):
            bNode = BNode.BNode()
            self.pyStore.add((bNode, REST, next))
            self.pyStore.add((bNode, FIRST, self.lookup(item)))
            next = bNode
        self.lists[node] = next
        return next

    def set(self, node):
        bNode = BNode.BNode()
        l = self.list(node)
        self.pyStore.add((bNode, OWLNS['oneOf'], l))
        return bNode

    def statements(self, formula):
        for var in formula.universals():
            self.pyStore.add_universal(self.lookup(var))
        for var in formula.existentials():
            if not isinstance(var, term.AnonymousNode):
                self.pyStore.add_existential(self.lookup(var))
        for statement in formula:
            self.triple(statement)
    
    def triple(self, statement):
        try:
            self.pyStore.add([self.lookup(item) for item in statement.spo()])
        except:
            raise

class FromPyStore(object):
    def __init__(self, formula, pyStore, parent=None):
        self.parent = parent
        self.formula = formula
        self.store = formula.store
        self.pyStore = pyStore
        self.bNodes = {}
        self.typeConvertors = [
            (Store.Store, self.subStore),
            (terms.Exivar, self.existential),
            (terms.Variable, self.variable),
            (terms.URIRef, self.URI),
            (BNode.BNode, self.BNode),
            (terms.Literal, self.literal)]
        self.stores = [
            (N3Loader.ClauseLoader, self.patterns),
            (N3Loader.N3Loader, self.facts_and_rules),
            (Store.Store, self.triples)]
        
    def lookup(self, node):
        for theType, function in self.typeConvertors:
            if isinstance(node, theType):
                return function(node)
        raise RuntimeError(`node` + '  ' + `node.__class__`)

    def run(self):
        node = self.pyStore
        for theType, function in self.stores:
            if isinstance(node, theType):
                return function(node)
        raise RuntimeError(`node` + '  ' + `node.__class__`)

    def URI(self, node):
        return self.formula.newSymbol(node)

    def variable(self, node):
        if self.pyStore.get_clause(node.name) is not None:
            return self.subStore(self.pyStore.get_clause(node.name))
        v = self.URI(node.name)
        self.parent.declareUniversal(v)
        return v

    def existential(self, node):
        if self.pyStore.get_clause(node.name) is not None:
            return self.subStore(self.pyStore.get_clause(node.name))
        v = self.URI(node.name)
        self.formula.declareExistential(v)
        return v
        
    def BNode(self, node):
        if self.pyStore.get_clause(node) is not None:
            return self.subStore(self.pyStore.get_clause(node))
        bNodes = self.bNodes
        if node not in bNodes:
            bNodes[node] = self.formula.newBlankNode(node)
        return bNodes[node]

    def literal(self, node):
        return self.formula.newLiteral(node, node.datatype or None, node.language or None)
    
    def subStore(self, node):
        f = self.formula.newFormula()
        self.__class__(f, node, self.formula).run()
        return f.close()

    def facts_and_rules(self, pyStore):
        patternMap = {}
        for nodeID in pyStore.list_clauses():
            patternMap[tuple(removedups(pyStore.get_clause(nodeID).patterns))] = pyStore.get_clause(nodeID)
        for fact in pyStore.facts:
            self.formula.add(
                self.lookup(fact.s),
                self.lookup(fact.p),
                self.lookup(fact.o))

        for rule in pyStore.rules:
            self.formula.add(
                self.subStore(patternMap[tuple(removedups(rule.lhs))]),
                self.store.implies,
                self.subStore(patternMap[tuple(removedups(rule.rhs))]))

    def patterns(self, pyStore):
        patternMap = {}
        for nodeID in pyStore.list_clauses():
            patternMap[tuple(removedups(pyStore.get_clause(nodeID).patterns))] = pyStore.get_clause(nodeID)
            
        for pattern in pyStore.patterns:
            if isinstance(pattern.s, terms.Rule):
                rule = pattern.s
                self.formula.add(
                    self.subStore(patternMap[tuple(removedups(rule.lhs))]),
                    self.store.implies,
                    self.subStore(patternMap[tuple(removedups(rule.rhs))]))
            else:
                self.formula.add(
                    self.lookup(pattern.s),
                    self.lookup(pattern.p),
                    self.lookup(pattern.o))

    def triples(self, pyStore):
        pass

if __name__ == '__main__':
    import sys
    #sys.path.append('/home/syosi')
    from swap import llyn
    #from pychinko.N3Loader import N3Loader
    store = llyn.RDFStore()
    from swap import webAccess
    f = webAccess.load(store, sys.argv[1])
    pyf = N3Loader.N3Loader()
    conv = ToPyStore(pyf)
    conv.statements(f)
    print "facts = " + ',\n'.join([repr(a) for a in pyf.facts])
    print "rules = " + ',\n'.join([repr(a) for a in pyf.rules])
    print '----'
    g = store.newFormula()
    reConv = FromPyStore(g, pyf)
    reConv.run()
    print g.close().n3String()
"""
