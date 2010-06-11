
#from pychinko.N3Loader import N3Loader
from terms import URI, Variable, Formula, Fact, Univar, Exivar
from helpers import removedups, keysToList, convertBNodeToFact, LOG_SEMANTICS, LOG_CONJUNCTION, LOG_CONCLUSION
from urllib import pathname2url
#from rdflib.BNode import BNode
#from rdflib.Graph import Graph
from copy import copy
import nodes
import sys
import os
import config
# add cwm builtins source to path
sys.path += [config.CWMSOURCE]
#print  sys.path
from  swap.llyn import RDFStore
from swap.term import Function, ReverseFunction, Term, CompoundTerm
from swap import cwm_string, cwm_list, cwm_math, cwm_os
#from pychinko.validation import Assert, Report, ValidationBuiltin
#from pychinko.N3Loader import formulas

from conclusion import Conclusion

store = RDFStore()

def builtinp(node):    
    return isinstance(node, Builtin) or  \
           isinstance(node, ALBuiltin)

def funcBuiltinp(node):
    if isinstance(node, Builtin):
        return node.pattern.p==LOG_SEMANTICS or node.pattern.p==LOG_CONJUNCTION or \
               node.pattern.p == LOG_CONCLUSION or (node.pattern.p in FuncBuiltins)

class Resource(object):
    def __init__(self):
        self.store = None

class CWMString(CompoundTerm):
    """I am a superfluous string holder used to pass in strings to CWM builtins"""
    def __init__(self, string, store):
        self.string = string
        self.store = store        

    def value(self):        
        return self.string

class Builtin(nodes.AlphaNode):
    """Placeholder for a builtin"""
    def __init__(self, pattern):
        self.URI = pattern.p
        self.pattern = pattern
        self.py_list = list() # python representation of rdf lists used in builtins,
                              #now this only represents cwmSubj, extend it for obj,too
        nodes.AlphaNode.__init__(self, pattern)                

    def evaluate(self, subj, obj):
        #print self.URI        
        if self.URI in CWMBuiltins or self.URI in FuncBuiltins:            
            """cwmSubj = subj
            cwmObj = obj"""
            cwmSubj = CWMString(subj, store)
            cwmObj = CWMString(obj, store)
            
            """print "subj:",cwmSubj.value()
            print "obj:",cwmObj.value()
            print "isinstance:", isinstance(cwmSubj,Term)"""
            
            # create the appropriate CWM builtin
            if self.URI in CWMBuiltins:
                builtin = CWMBuiltins[self.URI](cwmSubj, cwmObj)            
            else:
                builtin = FuncBuiltins[self.URI](cwmSubj, cwmObj)
            # unfortunately, the CWM builtin 'evaluate' function has the signature
            # (store, context, subj, subj_py, obj, obj_py) even though only 'subj'
            # and 'obj' are actually used in function body - so we must
            # pass in superfluous arguments
           
            #TODO: what kind of values to return here? convert to str or not?
            if isinstance(builtin, Function):                
                return str(builtin.evalObj(cwmSubj, False, False, False, False))
            elif isinstance(builtin, ReverseFunction):
                return str(builtin.evalSubj(cwmObj, False, False, False, False))
            else:                
                return builtin.eval(cwmSubj, cwmObj, False, False, False, False)
        elif self.URI in PychBuiltins:
            builtin = PychBuiltins[self.URI](self.pattern)
            return builtin.evaluate(subj, obj)
        else:
            raise exception.UnknownBuiltin(self, self.URI)
        
    def getInputNode(self):
        """I what node feeds input to me"""
        inputNode = None
        for b in self.betaNodes:
            #hack           
            if funcBuiltinp(b.lnode) or funcBuiltinp(b.rnode):                
                return b.lnode
            
            if builtinp(b.lnode) and not builtinp(b.rnode):                
                inputNode = b.rnode
            elif builtinp(b.rnode) and not builtinp(b.lnode):            
                inputNode = b.lnode                
            elif builtinp(b.rnode) and builtinp(b.lnode):
                if b.parents:
                    inputNode = b.parents[0]
                else:
                    inputNode = self            
            else:
                raise "No builtin to evaluate!"
            
                """ if isinstance(inputNode, pychinko.nodes.BetaNode) and inputNode.builtinInput:
                print "inputNode", inputNode.pattern
                print "bInput", inputNode.builtinInput
                inputNode.pattern = inputNode.builtinInput.pattern
                inputNode.ind = inputNode.ind                """
        return inputNode

    def indexKey(self):
        """I return the key according to which we build the bindings for the
        rows resulting builtin evaluation"""
        inputNode = self.getInputNode()
        
        if isinstance(inputNode, nodes.AlphaNode):            
            return removedups(inputNode.svars + inputNode.vars)
        elif isinstance(inputNode, nodes.BetaNode):            
            return inputNode.pattern
        
    
    def getbindings(self, row,useBuiltin=None):
        bindings = dict()
        key = self.indexKey()
        for i, v in enumerate(key):
            bindings[v] = row[i]
        return bindings

class Semantics(object):
    """An implementation of CWM's log:semantics"""
    def __init__(self, pattern):
        pass

    """retrieve the document at subj, parse it into a formula and store it in obj
       code similar to setup_and_run in main.py, uses N3loader"""
    def evaluate(self, subj, obj):
              
        if not subj.startswith('http://'):        
            subj = pathname2url(subj)        

        p = N3Loader()  
        p.parse(subj, format='n3')

        f = Formula(p)

        return f

class Conjunction(object):
    """An implementation of CWM's log:conjunction"""
    def __init__(self, pattern):
        pass
    
    """Take a list of  formulas as input in subject, merge them into a new formula and return the new one"""
    def evaluate(self, subj, obj):

        # need an N3 loader to get rules and facts first

        f = Formula()
        for conjunct in subj:
            f.patterns.extend(conjunct.patterns)               

        f.rules = list(f.getRules())
        f.facts = list(f.getFacts())
        
        
        return f

 



class notIncludes(object):
    """An implementation of CWM's log:notIncludes"""
    def __init__(self, pattern):
        pass
    
    def evaluate(self, subj, obj):
        inc = self.Includes()
        return not inc.evaluate(subj, obj)
    
            
class Includes(object):
    """An implementation of CWM's log:includes"""
    def __init__(self, pattern=None):
        pass

    """ log:includes implemented as simple matching:
         - for every fact in obj, see whether it's contained in subj
         - currently have only universal vars
         - ?a :y :z log:includes :x :y :z
         - no bindings passed for first-cut implementation
         - are there any bindings needed as input?"""         
    def evaluate(self, subj, obj):
        
#        print "Includes.."
#        print "subj: ", subj
#        print "obj: ", obj       
        return self.testIncludes(subj.patterns, obj.patterns)
        

    """does lhs log:include rhs?"""
    def matchLiteral(self, lhs, rhs):
        if isinstance(rhs,Univar):
            print "cwm does not allow universals in object of log:includes", rhs
            return False
        elif isinstance(rhs,Exivar):
            return True
        elif isinstance(lhs, Univar):
            return True
        elif isinstance(lhs, Exivar):
            return False
        else: #no vars, just compare facts
            return rhs == lhs
                    
    
    """ I match a triple with subject formula. The triple does not contain any nested graphs"""
    def matchTriple(self, sFormula, triple):        
                            
        #eliminate triples in sFormula that contain formulas  themselves
        l = [ item for item in sFormula if not isinstance(item.s, Formula) and not isinstance(item.o, Formula) ]
        

        # match the triple
        l = [item for item in sFormula if self.matchLiteral(item.s, triple.s) and self.matchLiteral(item.p, triple.p) and
                                         self.matchLiteral(item.o, triple.o)]

        
        return len(l) > 0 
                
    """ first compare the top level statements, then the nested formulas recursively """
    def testIncludes(self, sFormula, oFormula):        
        for triple in oFormula:            
            if  not isinstance(triple.s, Formula)  and  not isinstance(triple.o, Formula):
                if not self.matchTriple(sFormula, triple):
                    return False
                
            elif isinstance(triple.s, Formula) and isinstance(triple.o, Formula):
                # {} _ {} 
                filtered = [item for item in sFormula if isinstance(item.s, Formula) and
                                                                       self.matchLiteral(item.p, triple.p)]
                                                          
                filtered = [item for item in filtered if isinstance(item.o, Formula)]
                                
                 
                #go through the filtered triples and invoke testIncludes recursively
                #if there's at least one match return true
                matched = False
                for i in filtered:
                     if (self.testIncludes(i.s.patterns, triple.s.patterns) and
                            self.testIncludes(i.o.patterns, triple.o.patterns)):
                        matched = True                                                                                           
                        break
                    
                if not matched:
                    return False

            elif isinstance(triple.s, Formula):
                
                # {} _ _                 
                filtered = [item for item in sFormula if isinstance(item.s, Formula) and
                                                          self.matchLiteral(item.p, triple.p) and self.matchLiteral(item.o, triple.o)]                
                
                matched = False
                for i in filtered:                    
                    if self.testIncludes(i.s.patterns, triple.s.patterns):
                        matched = True
                        break
                    
                if not matched:
                    return False                
                
            elif isinstance(triple.o, Formula):
                # _ _ {}
                filtered = [item for item in sFormula if isinstance(item.o, Formula) and
                                                          self.matchLiteral(item.p, triple.p) and self.matchLiteral(item.s, triple.s)]        
             
                matched = False
                for i in filtered:
                    if self.testIncludes(i.o.patterns, triple.o.patterns):
                        matched = True
                        break
                    
                if not matched:
                    return False

        #print "returning true..."        
        return True
                    

class PelletQuery(object):
    """Ship RDQL queries to Pellet and return the result"""
    def __init__(self, kb):
        self.kb = kb
        self.pellet = self.getPellet() 
        self.runCmd = self.pellet + ' -classify N3 -realize -inputFormat N3 -inputFile ' + self.kb

    def getPellet(self):
        """I read Pellet's location from ../config/pellet.conf"""
        import ConfigParser
        config = ConfigParser.ConfigParser()
        config.readfp(open('config/pellet.conf'))
        for name, value in config.items("Pellet"):
            if name == 'pelletcmd':
                return value

    def queryString(self, query):
        """I ship an RDQL query contained in a string to Pellet and return the result"""
        queryfp = open('query', 'w')
        queryfp.write(query + '\n')
        queryfp.close()
        query = self.runCmd + ' -queryFile query'
        pelletfp = os.popen(query, 'r')
        print pelletfp.read()

# Implementation of AL-log.
#
# Our rules are of the same form as before.  To gain the ALC expressivity
# of class instance testing, we define the builtin :instanceOf whose object
# argument is a class name (a string.)
#
# We still need to figure out how to have class expressions, e.g. :instanceOf (C v D).

class ALBuiltin(Builtin):
    def __init__(self, pattern):
        Builtin.__init__(self, pattern)

    def evaluate(self, subj, obj):
        if self.URI in PychBuiltins:
            print "Known builtin (%s)" %(self.URI)
            print "Subj: ", subj
            print "Obj: ", obj
            builtin = PychBuiltins[self.URI](self.pattern)
            return builtin.evaluate(subj, obj)
        else:
            print "Unknown builtin (%s)" %(self.URI)
            return None

class instanceOf(ALBuiltin):
    def __init__(self, pattern):
        ALBuiltin.__init__(self, pattern)

    def evaluate(self, subj, obj):
        print "Evaling s->(%s) o->(%s)" %(subj, obj)
        pellet = PelletQuery("allogtests/allog-facts.n3")
        pellet.queryString("SELECT ?x, ?y\nWHERE (?x, <http://cwmTest/parent>, ?z)")

# CWM has a store.internURI() method which already does this, but tapping into
# will involve using their RDFStore (which can be rdflib, I believe) but it's
# nontrivial so I'm recreating it here until full integration
## These are builtins that are called directly from the CWM source
CWMBuiltins = {

               URI('http://www.w3.org/2000/10/swap/math#negation'):
               cwm_math.BI_negation,
               URI('http://www.w3.org/2000/10/swap/math#absoluteValue'):               
               cwm_math.BI_absoluteValue,
               URI('http://www.w3.org/2000/10/swap/math#rounded'):
               cwm_math.BI_rounded,               
               URI('http://www.w3.org/2000/10/swap/list#in'):
               cwm_list.BI_in,

    
               URI('http://www.w3.org/2000/10/swap/math#notGreaterThan'):
               cwm_math.BI_notGreaterThan,
               URI('http://www.w3.org/2000/10/swap/math#notLessThan'):
               cwm_math.BI_notLessThan,

               URI('http://www.w3.org/2000/10/swap/math#equalTo'):
               cwm_math.BI_equalTo,
               URI('http://www.w3.org/2000/10/swap/math#notEqualTo'):
               cwm_math.BI_notEqualTo,

               URI('http://www.w3.org/2000/10/swap/math#memberCount'):
               cwm_math.BI_memberCount,
                              
               #cwm relational string builtins
               URI('http://www.w3.org/2000/10/swap/string#greaterThan'):
               cwm_string.BI_GreaterThan,
               URI('http://www.w3.org/2000/10/swap/string#lessThan'):
               cwm_string.BI_LessThan,
               URI('http://www.w3.org/2000/10/swap/math#greaterThan'):
               cwm_math.BI_greaterThan,
               URI('http://www.w3.org/2000/10/swap/math#lessThan'):
               cwm_math.BI_lessThan,               
               URI('http://www.w3.org/2000/10/swap/string#contains'):
               cwm_string.BI_Contains,
               URI('http://www.w3.org/2000/10/swap/string#startsWith'):
               cwm_string.BI_StartsWith,
               URI('http://www.w3.org/2000/10/swap/string#endsWith'):
               cwm_string.BI_EndsWith,
               URI('http://www.w3.org/2000/10/swap/string#greaterThan'):               
               cwm_string.BI_GreaterThan,
               URI('http://www.w3.org/2000/10/swap/string#notGreaterThan'):
               cwm_string.BI_NotGreaterThan,
               URI('http://www.w3.org/2000/10/swap/string#lessThan'):
               cwm_string.BI_LessThan,
               URI('http://www.w3.org/2000/10/swap/string#notLessThan'):
               cwm_string.BI_NotLessThan,
               
               URI('http://www.w3.org/2000/10/swap/string#notEqualIgnoringCase'):
               cwm_string.BI_notEqualIgnoringCase}

PychBuiltins = {
#                 URI('http://www.mindswap.org/~katz/pychinko/builtins#assert'):
#                Assert,
#                URI('http://www.mindswap.org/~katz/pychinko/builtins#report'):
#                Report,
                URI('http://www.mindswap.org/~katz/pychinko/builtins#instanceOf'):
                instanceOf,
                URI('http://www.w3.org/2000/10/swap/log#includes'):
                Includes,
                URI('http://www.w3.org/2000/10/swap/log#semantics'):
                Semantics,
                URI('http://www.w3.org/2000/10/swap/log#conjunction'):
                Conjunction,
                URI('http://www.w3.org/2000/10/swap/log#conclusion'):
                Conclusion
                }


FuncBuiltins = {URI('http://www.w3.org/2000/10/swap/math#sum'):
               cwm_math.BI_sum,
               URI('http://www.w3.org/2000/10/swap/math#sum'):
               cwm_math.BI_sum,
               URI('http://www.w3.org/2000/10/swap/math#difference'):
               cwm_math.BI_difference,
               URI('http://www.w3.org/2000/10/swap/math#product'):
               cwm_math.BI_product,
               URI('http://www.w3.org/2000/10/swap/math#integerQuotient'):
               cwm_math.BI_integerQuotient,
               URI('http://www.w3.org/2000/10/swap/math#quotient'):
               cwm_math.BI_quotient,
               URI('http://www.w3.org/2000/10/swap/math#remainder'):
               cwm_math.BI_remainder,
               URI('http://www.w3.org/2000/10/swap/math#exponentiation'):
               cwm_math.BI_exponentiation,
               
               URI('http://www.w3.org/2000/10/swap/math#sumOf'):
               cwm_math.BI_sumOf,
               URI('http://www.w3.org/2000/10/swap/math#factors'):
               cwm_math.BI_factors,
               URI('http://www.w3.org/2000/10/swap/math#bit'):
               cwm_math.BI_bit,
               URI('http://www.w3.org/2000/10/swap/math#quotientOf'):
               cwm_math.BI_quotientOf,
               URI('http://www.w3.org/2000/10/swap/math#remainderOf'):               
               cwm_math.BI_remainderOf,
               URI('http://www.w3.org/2000/10/swap/math#exponentiationOf'):
               cwm_math.BI_exponentiationOf,

               URI('http://www.w3.org/2000/10/swap/math#negation'):
               cwm_math.BI_negation,
               URI('http://www.w3.org/2000/10/swap/math#absoluteValue'):               
               cwm_math.BI_absoluteValue,
               URI('http://www.w3.org/2000/10/swap/math#rounded'):
               cwm_math.BI_rounded,

                #string builtins
               URI('http://www.w3.org/2000/10/swap/string#concatenation'):
               cwm_string.BI_concatenation, 
               
               
                }

logimplies = (URI, 'http://www.w3.org/2000/10/swap/log#logimplies')

def getPatternType(pred):
        
    if pred in PychBuiltins or pred in CWMBuiltins or pred in FuncBuiltins:
        return Builtin    
#     elif pred in CWMBuiltins:
#         print "RETURNING ---> ", CWMBuiltins[pred]
#         return CWMBuiltins[pred]
#     else:
#         return None


