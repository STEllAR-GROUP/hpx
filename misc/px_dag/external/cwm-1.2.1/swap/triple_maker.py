"""Triple Maker

$Id: triple_maker.py,v 1.10 2005/01/21 20:54:04 syosi Exp $
Explanation of the API

the functions are addNode(),addNode(), addNode() endStatement() to add a triple
never call addNode() yourself
addSymbol() and addLiteral() are there for those
if a Node is a compound structure (like a formula) then call beginFormula(),
add all of the structure to the substructure, then all endFormula() and that
will call addNode() for you.

For your convinience, if you call IsOf() before adding the predicate, it will
reverse the subject and object in the final triple
Also for your convience, you CAN call addNode() with None as the node,
and it will just leave that as the previous node in that position.
Note that calling addNode() with None as the first triple in a formula or
bNode is an error, and will be flagged when you try to endStatement()

"""


import diag  # problems importing the tracking flag, and chatty_flag must be explicit it seems diag.tracking
from diag import progress, verbosity
from term import BuiltIn, LightBuiltIn, \
    HeavyBuiltIn, Function, ReverseFunction, \
    Literal, Symbol, Fragment, FragmentNil, Term,\
    CompoundTerm, List, EmptyList, NonEmptyList, AnonymousNode

import RDFSink
N3_forSome_URI = RDFSink.forSomeSym
N3_forAll_URI = RDFSink.forAllSym

NOTHING   = -1
SUBJECT   = 0
PREDICATE = 1
OBJECT    = 2

FORMULA = 0
LIST    = 1
ANONYMOUS = 2

NO = 0
STALE = 1
FRESH = 2

def swap(List, a, b):
    q = List[a]
    List[a] = List[b]
    List[b] = q

class TripleMaker:
    """This is the example of the new interface.
    It will convert from the abstract interface into formulas, lists
    and triples



    """
    def __init__(self, formula=None, store=None):
        if formula is None:
            formula = store.newFormula
        if store is None:
            store = formula.store
        self.formulas = [formula]
        self.store = store
        self.forSome = formula.newSymbol(N3_forSome_URI)
        self.forAll  = formula.newSymbol(N3_forAll_URI)

    def start(self):
        self._parts = [NOTHING]
        self._triples = [[None, None, None]]
        self.lists = []
        self._modes = [FORMULA]
        self.bNodes = []
        self.addedBNodes = [{}]
        self._predIsOfs = [NO]
        self._pathModes = [False]
        self.store.startDoc()

    def end(self):
        assert len(self.formulas) == 1
        assert len(self.lists) == 0
        self.store.endDoc(self.formulas[0])
        return self.formulas[0]

    def addNode(self, node):
        if self._modes[-1] == ANONYMOUS and node is not None and self._parts[-1] == NOTHING:
            raise ValueError('You put a dot in a bNode')
        if self._modes[-1] == FORMULA or self._modes[-1] == ANONYMOUS:
            self._parts[-1] = self._parts[-1] + 1
            if self._parts[-1] > 3:
                raise ValueError('Try ending the statement')
            if node is not None:
                #print node, '+', self._triples, '++', self._parts
                self._triples[-1][self._parts[-1]] = node
                if self._parts[-1] == PREDICATE and self._predIsOfs[-1] == STALE:
                    self._predIsOfs[-1] = NO
        if self._modes[-1] == ANONYMOUS and self._pathModes[-1] == True:
            self.endStatement()
            self.endAnonymous()
        elif self._modes[-1] == LIST:
            self.lists[-1].append(node)

    def IsOf(self):
        self._predIsOfs[-1] = FRESH

    def checkIsOf(self):
        return self._predIsOfs[-1]

    def forewardPath(self):
        if self._modes[-1] == LIST:
            a = self.lists[-1].pop()
        else:
            a = self._triples[-1][self._parts[-1]]
            self._parts[-1] = self._parts[-1] - 1
        self.beginAnonymous()
        self.addNode(a)
        self._predIsOfs[-1] = FRESH
        self._pathModes[-1] = True
        
    def backwardPath(self):
        if self._modes[-1] == LIST:
            a = self.lists[-1].pop()
        else:
            a = self._triples[-1][self._parts[-1]]
            self._parts[-1] = self._parts[-1] - 1
        self.beginAnonymous()
        self.addNode(a)
        self._pathModes[-1] = True

    def endStatement(self):
        if self._parts[-1] == SUBJECT:
            pass
        else:
            if self._parts[-1] != OBJECT:
                raise ValueError('try adding more to the statement' + `self._triples`)
            formula = self.formulas[-1]

            if self._pathModes[-1]:
                swap(self._triples[-1], PREDICATE, OBJECT)
            if self._predIsOfs[-1]:
                swap(self._triples[-1], SUBJECT, OBJECT)
                self._predIsOfs[-1] = STALE 
            subj, pred, obj = self._triples[-1]

            if subj == '@this':
                if pred == self.forSome:
                    formula.declareExistential(obj)
                elif pred == self.forAll:
                    formula.declareUniversal(obj)
                else:
                    raise ValueError("This is useless!")
            else:
                #print 'I got here!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                formula.add(subj, pred, obj)
            if self._predIsOfs[-1]:
                swap(self._triples[-1], SUBJECT, OBJECT)
        self._parts[-1] = NOTHING
        if self._modes[-1] == ANONYMOUS and self._pathModes[-1]:
            self._parts[-1] = SUBJECT

    def addLiteral(self, lit, dt=None, lang=None):
        if dt:
            if dt[:2] == '_:':
                if dt not in self.addedBNodes[-1]:
                    a = self.formulas[-1].newBlankNode()
                    self.addedBNodes[-1][dt] = a
                    dt = a
                else:
                    dt = self.addedBNodes[-1][dt]
            else:
                dt = self.store.newSymbol(dt)
        a = self.store.intern(lit, dt, lang)
        self.addNode(a)

    def addSymbol(self, sym):
        a = self.store.newSymbol(sym)
        self.addNode(a)
    
    def beginFormula(self):
        a = self.store.newFormula()
        self.formulas.append(a)
        self.addedBNodes.append({})
        self._modes.append(FORMULA)
        self._triples.append([None, None, None])
        self._parts.append(NOTHING)
        self._predIsOfs.append(NO)
        self._pathModes.append(False)

    def endFormula(self):
        if self._parts[-1] != NOTHING:
            self.endStatement()
        a = self.formulas.pop().close()
        self.addedBNodes.pop()
        self._modes.pop()
        self._triples.pop()
        self._parts.pop()
        self.addNode(a)
        self._predIsOfs.pop()
        self._pathModes.pop()

    def beginList(self):
        a = []
        self.lists.append(a)
        self._modes.append(LIST)
        self._parts.append(NOTHING)

    def endList(self):
        a = self.store.newList(self.lists.pop())
        self._modes.pop()
        self._parts.pop()
        self.addNode(a)

    def addAnonymous(self, Id):
        """If an anonymous shows up more than once, this is the
        function to call

        """
        if Id not in self.addedBNodes[-1]:
            a = self.formulas[-1].newBlankNode()
            self.addedBNodes[-1][Id] = a
        else:
            a = self.addedBNodes[-1][Id]
        self.addNode(a)
        
    
    def beginAnonymous(self):
        a = self.formulas[-1].newBlankNode()
        self.bNodes.append(a)
        self._modes.append(ANONYMOUS)
        self._triples.append([a, None, None])
        self._parts.append(SUBJECT)
        self._predIsOfs.append(NO)
        self._pathModes.append(False)
        

    def endAnonymous(self):
        if self._parts[-1] != NOTHING:
            self.endStatement()
        a = self.bNodes.pop()
        self._modes.pop()
        self._triples.pop()
        self._parts.pop()
        self._predIsOfs.pop()
        self._pathModes.pop()
        self.addNode(a)

    def declareExistential(self, sym):
        formula = self.formulas[-1]
        a = formula.newSymbol(sym)
        formula.declareExistential(a)

    def declareUniversal(self, sym):
        formula = self.formulas[-1]
        a = formula.newSymbol(sym)
        formula.declareUniversal(a)

    def addQuestionMarkedSymbol(self, sym):
        formula = self.formulas[-2]
        a = formula.newSymbol(sym)
        formula.declareUniversal(a)
        self.addNode(a)

    def bind(self, prefix, uri):
        if prefix == "":
            self.store.setDefaultNamespace(uri)
        else:
            self.store.bind(prefix, uri)

    
        
        
    

