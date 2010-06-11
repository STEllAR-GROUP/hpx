import time, sys
from exceptions import StandardError
#from rdflib.URIRef import URIRef
#from rdflib.BNode import BNode
#from rdflib import Namespace
from types import IntType

from swap.myStore  import Namespace

LOG = Namespace("http://www.w3.org/2000/10/swap/log#")


class URIRef(unicode):
    def __new__(cls, value, base=None):
        if base is not None:
            value = urljoin(base, value, allow_fragments=1)
        #if normalize and value and value != normalize("NFC", value):
        #    raise Error("value must be in NFC normalized form.")
        return unicode.__new__(cls,value)
        
        
#XXX: This now accepts URIRef's too since that is what
#XXX: rdflib.constants uses
class URI(URIRef):
        
    def __init__(self, s):        
        URIRef.__init__(self, unicode(s))        
        self.name = unicode(s)
#        unicode.__new__(cls,value)

#     def __cmp__(self, other):
#         return unicode(self) is unicode(other)

    def __repr__(self): return """%s""" % (self.name)


class Formula:

    #todo finish here!
    def __init__(self, n3l=None):
        if n3l:
            self.patterns = list(n3l.patterns())
            self.id = n3l.identifier
            self.rules = list(n3l.rules())
            self.facts = list(n3l.facts())
        else:
            self.patterns = list()
            # get new bnode value here
            tmp =  BNode()
            self.id = tmp.n3()
            print "_________ID:", self.id
            self.rules = list()
            self.facts = list()

    # version of facts() and rules() butthis time input is a terms.formula, not rdflib.graph
    # these are used in log:conjunction
    def getFacts(self):
       for s, p, o in self.patterns:
           if p!= LOG.implies and not isinstance(s, Variable) and not isinstance(o, Variable): # and not isinstance(s, BNode) and not isinstance(o, BNode):               
               yield Fact(s,p,o)

    def getRules(self):
       for s, p, o in self.patterns:
           if p == LOG.implies:               
               lhs = list(s.patterns)
               rhs = list(o.patterns)
               yield Rule(lhs, rhs, (s, p, o))
    
       
    def add(self, triple):
        self.patterns.append(triple)

                
    def __repr__(self):
        if self.patterns == []:
            return "{}"
        
        s = "Formula ID:%s" % self.id
        s += "\nFormula(%s)\n"  % self.patterns
        
               
        return s
    

class Variable(object):
    def __init__(self, s):
        """I am a variable. My only argument is a string identifier."""
        self.name = unicode(s)

    def __repr__(self): return """?%s""" % (self.name)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name
        return False

    def __neq__(self, other): return not(self.__eq__(other))

class Exivar(Variable):
    """I am a space holder for an N3 existential variable."""
    def __init__(self, s):
        Variable.__init__(self, s)

    def __str__(self):
        return "_:" + self.name

class Univar(Variable):
    """I am a space holder for an N3 universal variable."""
    def __init__(self, s):
        Variable.__init__(self, s)
    
    
class Triple:
    label = "Triple"
    def __init__(self, s, p, o):
        """I am an RDF 3-tuple: (subject, predicate, object)."""
        self._s = s; self._o = o; self._p = p
        self.spo = [s, p, o]
                
    def getS(self): return self._s
    def getP(self): return self._p
    def getO(self): return self._o
    def getT(self): return tuple((self._s, self._p, self._o))
    
    s = property(getS, None, None, "Subject")
    p = property(getP, None, None, "Predicate")
    o = property(getO, None, None, "Object")
    t = property(getT, None, None, "Triple")

    def index(self, var):
        return self.spo.index(var)

    def __getitem__(self, index):
        if type(index) is not IntType:
            raise TypeError, "index must be an integer."
        else:
            return self.spo[index]

    def __eq__(self, other):
        return not self.__ne__(other)

    def __ne__(self, other):
        if not isinstance(other, Triple):
            return True
        if self.s != other.s:
            return True
        elif self.p != other.p:
            return True
        elif self.o != other.o:
            return True
        else:
            return False

    def __repr__(self): return """%s%s""" % (self.label, self.t)

class Fact(Triple):
    label = "Fact"
    ######Change this to a "Fact Factory!"
    def __init__(self, s, p, o):
        """I am a Fact. I take grounded subject, predicate, and object arguments."""
        if isinstance(s, Variable) or isinstance(p, Variable) \
               or isinstance(o, Variable) or isinstance(s, list) or isinstance(p, list) \
               or isinstance(o, list):            
            raise TypeError, "Facts are patterns with three grounded values."
        Triple.__init__(self, s, p, o)

    def __hash__(self):
        return hash((self._s, self._p, self._o))

    def __repr__(self):
        return """Fact(%s, %s, %s)""" %(self.s, self.p, self.o)

class Pattern(Triple):
    label = "Pattern"
    def __init__(self, s, p, o):
        """I am a Pattern. I take subject, predicate, object arguments"""
        self.spo = (s, p, o)
        Triple.__init__(self, s, p, o)

    def index(self, var):
        return list(self.spo).index(var)

    def __hash__(self):
        return hash((self._s, self._p, self._o))

    def __len__(self):
        return len(self.spo)

    def noneBasedPattern(self):
        """I return a pattern that does not distinguish between variables; that is, I keep the constants as is and replace any variable with a 'None'"""
        nonePattern = list()
        for i in self.spo:
            if isinstance(i, Variable): 
                nonePattern.append(None)
            else:
                nonePattern.append(i)
        return tuple(nonePattern)

class Rule(object):
    label = "Rule"
    def __init__(self, lhs, rhs, (subj,pred,obj)):
        """I am a Rule. My arguments are left and right hand sides, which
        are arbitrary-lengthened lists of instances of Pattern."""
        for i in lhs:
            if not isinstance(i, Pattern):
                raise TypeError, \
                      "The members of a Rule's LHS must be Patterns (see item: %s)." % (i)
            else: continue
        for i in rhs:
            if not isinstance(i, Pattern):
                raise TypeError, \
                      "The members of a Rule's RHS must be Patterns (see item: %s)." % (i)
            else: continue
        self.__lhs = lhs
        self.__rhs = rhs
        self.__ts = time.time() #I think this will break on Windows...
        self.__triple = (subj,pred,obj) #this is so we can identify the lhs and rhs easily

    def getRuleTriple(self): return self.__triple
    def getTs(self): return self.__ts
    def setTs(self, val): self.__ts = val
    def getLhs(self): return self.__lhs
    def setLhs(self, val): self.__lhs = val
    def getRhs(self): return self.__rhs
    def setRhs(self, val): self.__rhs = val
    
    timestamp = property(getTs, setTs, None, None)
    lhs = property(getLhs, setLhs, None, None)
    rhs = property(getRhs, setRhs, None, None)
    
    def __repr__(self):
        return """%s(%s =>\n\t%s :timestamp %s)""" % (self.label, self.lhs,
                                                      self.rhs, self.timestamp)

class BuiltinRule(Rule):
    def __init__(self, lhs, rhs, builtin):
        Rule.__init__(self, lhs, rhs)
        self.builtin = builtin

class TokenConstructionError(StandardError): pass

class Token(object):
    def __init__(self, tag, fact):
        if isinstance(tag, Tag):
            self.tag = tag
        else:
            raise TokenConstructorError
        self.fact = fact

    def __repr__(self):
        return """Token(%s, %s)""" % (self.tag, self.fact)

    def __eq__(self, other):
        if self.tag == other.tag:
            if self.fact == other.fact:
                return True
            return False
        return False
    
#abstract base class
class Tag(object):

    def __repr__(self):
        return """%s""" % (self.label)

    def __eq__(self, other):
        if type(self) == type(other): return True
        return False
    
class Add(Tag): label = "+"
class Delete(Tag): label = "-"

def makeToken(t, l):
    if t == "+": tag = Add()
    elif t == "-": tag = Delete()
    else:
        raise TokenConstructionError
    return Token(tag, l)
