#!/usr/bin/env python
"""sparql2cwm

This is meant to be used with a sparql.n3 based SPARQL parser, to add the query to cwm

$Id: sparql2cwm.py,v 1.24 2007/08/06 16:13:56 syosi Exp $
"""

from set_importer import Set
import uripath
from term import Term, CompoundTerm
from formula import Formula
import diag
from why import BecauseOfData

from cwm_sparql import SPARQL_NS

knownFunctions = {}

def value(self, newVal=None):
    if newVal is not None:
        self[0] = newVal
    return self[0]
verbose = value.__get__([0])
reason2 = value.__get__([0])

def abbr(prodURI):
   return prodURI.split('#').pop()

class typedThing(unicode):
    def __new__(cls, val, retType=None, ArgTypes=[], trueOnError=False):
        ret = unicode.__new__(cls, val)
        ret.retType = retType
        ret.argTypes = ArgTypes
        ret.trueOnError = trueOnError
        return ret

    def __call__(self):
        return unicode(self) + '__' + unicode(self.retType) + '__' + {False:'', True:'lenient__'}[self.trueOnError] + '_'.join(self.argTypes)

def getType(ex):
    if isinstance(ex, typedThing):
        return ex.retType
    return None

def getTrueOnError(ex):
    if isinstance(ex, typedThing):
        return ex.trueOnError
    return False

ORED_AND = typedThing('And', 'boolean', trueOnError=True)
AND = typedThing('And', 'boolean')
OR = typedThing('Or', 'boolean')
ORED_NOT = typedThing('Not', 'Boolean', trueOnError=True)
NOT = typedThing('Not', 'boolean')

class andExtra(tuple):
    def __new__(cls, val, extra):
        ret = tuple.__new__(cls, val)
        ret.extra = extra
        return ret

    def __repr__(self):
        return repr(tuple(self)) + '+' + repr(self.extra)

def getExtra(ex):
    if isinstance(ex, andExtra):
        return ex.extra
    return []

class multimap(dict):
    """ A class to handle where each element maps to a set of elements

It would perhaps be simpler to have used dict.setdefault(), but this has the
advantage of merging instead of nesting sets

"""
    class innerSet(Set):
        pass
    def __init__(self, olddict={}, **values):
        self.update(olddict)
        self.update(values)
    def __setitem__(self, key, value):
        if isinstance(value, tuple):
            raise RuntimeError(key, value)
        if not key in self:
            dict.__setitem__(self, key, self.innerSet())
        if isinstance(value, self.innerSet):
            self[key].update(value)
        elif value:
            self[key].add(value)
    def update(self, other={}, **values):
        if values:
            self.update(values)
        for key, val in other.iteritems():
            self[key] = val
    def translate(self, fromkey, tokey):
        if fromkey not in self:
            return
        m = self[fromkey]
        del self[fromkey]
        self[tokey] = m
    def __add__(self, other):
        k = self.__class__()
        k.update(self)
        k.update(other)
        return k
    def _fromTuples(cls, iterator):
        m = cls()
        for key, val in iterator:
            m[key] = val
        return m
    fromTuples = classmethod(_fromTuples)


def makeTriple(subj, pred, obj, safeVersion=False):
    if safeVersion:
        store = pred[1].store
        typeErrorIsTrue = store.newSymbol(SPARQL_NS + '#typeErrorIsTrue')
        return ('Triple', (('Literal', store.intern(1)), ('predicateList',
                                         [(('symbol', typeErrorIsTrue), ('objectList',
                                                  [('formula', TripleHolder((subj[1], pred[1], obj[1])))]))])))
    return ('Triple', (subj, ('predicateList',
                                         [(pred, ('objectList',
                                                  [obj]))])))

def makeSafeVal(val, (subj, pred, obj), safeVersion=False):
    if safeVersion:
        store = pred[1].store
        typeErrorReturner = store.newSymbol(SPARQL_NS + '#typeErrorReturner')
        replacement = ('Literal', store.intern(1))
        if subj==val:
            subj=replacement
            return makeTriple(val, ('formula', TripleHolder((subj[1], pred[1], obj[1]))), ('symbol', typeErrorReturner))
        if obj==val:
            obj=replacement
            return makeTriple(('formula', TripleHolder((subj[1], pred[1], obj[1]))), ('symbol', typeErrorReturner), val)
    return makeTriple(subj, pred, obj)
        

def makeTripleObjList(subj, pred, obj):
    return ('Triple', (subj, ('predicateList',
                                         [(pred, ('objectList',
                                                  obj))])))

def normalize(expr):
    """ The mapping from SPARQL syntax to n3 is decidedly nontrivial
at this point, we have done the first step, building an AST that is (relatively) independant of all of the oddities that
you get from the grammar

Normalize does a couple of top down transforms on the tree. The first is a type determiner; it determines what
needs to be coerced to a boolean. The second does boolean logic and pushes ``not''s all the way in.

After normalize, the last bottom up step to generate the n3 can be done.
    """
    if verbose(): print expr
    step1 = Coerce()(expr)
    return NotNot()(step1)

class Coerce(object):
    def __init__(self):
        self.k = 0
    def __call__(self, expr, coerce=True):
        try:
            if verbose(): print '  ' * self.k, expr, coerce
            self.k = self.k + 1
            if expr[0] in ('Var', 'Literal', 'Number', 'String', 'symbol'):
                ww = self.atom(expr, coerce)
            elif expr[0] in ('subtract', 'add', 'multiply', 'divide', 'lang', 'datatype'):
                ww = self.on_math(expr, coerce)
            elif expr[0] in ('less', 'equal', 'notEqual', 'greater', 'notLess', 'notGreater'):
                ww = self.on_pred(expr, coerce)
            else:
                ww = getattr(self, 'on_' + expr[0])(expr, coerce)
            self.k = self.k - 1
            if verbose(): print '  ' * self.k, '/', ww
            return ww
        except AttributeError:
             raise RuntimeError("COERCE why don't you define a %s function, to call on %s?" % ('on_' + expr[0], `expr`))

    def on_function(self, p, coerce):
        if coerce:
            return ('BoolVal', self(p, False))
        return [p[0], p[1]] + [self(q, False) for q in p[2:]]

    def on_Or(self, p, coerce):
        if len(p) == 2:
            return self(p[1], coerce)
        return [p[0]] + [self(q, True) for q in p[1:]]

    def on_And(self, p, coerce):
        if len(p) == 2:
            return self(p[1], coerce)
        return [p[0]] + [self(q, True) for q in p[1:]]

    def on_BoolVal(self, p, coerce):
        if coerce:
            return [p[0], self(p[1], False)]
        return self(p[1], False)

    def atom(self, p, coerce):
        if coerce:
            return ('BoolVal', p)
        return p

    def on_math(self, p, coerce):
        retVal = [p[0]] + [self(q, False) for q in p[1:]]
        if coerce:
            return ('BoolVal', retVal)
        return retVal

    def on_Boolean(self, p, coerce):
        if coerce:
            return ('BoolVal', p)
        return p
    def on_Bound(self, p, coerce):
        return p

    def on_Regex(self, p, coerce):
        return [p[0]] + [self(q, False) for q in p[1:]]

    def on_pred(self, p, coerce):
        return [p[0]] + [self(q, False) for q in p[1:]]

    def on_Not(self, p, coerce):
        return [p[0], self(p[1], True)]

    def on_isURI(self, p, coerce):
        return [p[0], self(p[1], False)]
    def on_isLiteral(self, p, coerce):
        return [p[0], self(p[1], False)]
    def on_isBlank(self, p, coerce):
        return [p[0], self(p[1], False)]

class NotNot(object):
    """ This class is used to figure out the inverses of all SPARQL boolean tests, and invert all functions

"""
    inverse_operators = {'less' : 'notLess',
                         'greater' : 'notGreater',
                         'notLess' : 'less',
                         'notGreater' : 'greater',
                         'equal' : 'notEqual',
                         'notEqual': 'equal',
                         'isURI' : 'isNotURI',
                         'isNotURI' : 'isURI',
                         'isLiteral' : 'isNotLiteral',
                         'isNotLiteral' : 'isLiteral',
                         'isBlank' : 'isNotBlank',
                         'isNotBlank' : 'isBlank',
##                         'Not', 'BoolVal',
##                         'BoolVal', 'Not',
                         'Bound': 'notBound' }
                         
    def __init__(self):
        self.k = 0
    
    def __call__(self, expr, inv=False, Ored=False):
        try:
            if verbose(): print '  ' * self.k, expr, inv
            self.k = self.k + 1
            if not isinstance(expr, (list, tuple)):
                return expr
            if expr[0] in self.inverse_operators:
                ww = self.expr(expr, inv, Ored)
            elif expr[0] in ('Var', 'Literal', 'Number', 'subtract', 'add', 'datatype',
                             'multiply', 'divide', 'String', 'symbol', 'function', 'lang'):
                ww = self.atom(expr, inv, Ored)
            else:
                ww = getattr(self, 'on_' + expr[0])(expr, inv, Ored)
            self.k = self.k - 1
            if verbose(): print '  ' * self.k, '/', ww
            return ww
        except AttributeError:
             raise RuntimeError("NOTNOT why don't you define a %s function, to call on %s?" % ('on_' + expr[0], `expr`))

    def expr(self, p, inv, ored):
        if inv:
            return [typedThing(self.inverse_operators[p[0]], 'boolean', trueOnError=ored)] + [self(q, False, ored) for q in p[1:]]
        return [typedThing(p[0], trueOnError=ored)] + [self(q, False, ored) for q in p[1:]]

    def atom(self, p, inv, ored):
        if ored:
            p = [typedThing(p[0], getType(p[0]), trueOnError=True)] + [self(q, False, True) for q in p[1:]]
        if inv and ored:
            return (ORED_NOT, p)
        if inv:
            return (NOT, p)
##        if ored:
##            return ('typesafe', p)
        return p

    def on_Not(self, p, inv, ored):
        if inv:
            return self(p[1], False, ored)
        return self(p[1], True, ored)

    def on_Regex(self, p, inv, ored):
        if inv:
            return [typedThing('notMatches', trueOnError=ored)] + [self(q, False, ored) for q in p[1:]]
        return [typedThing(p[0], trueOnError=ored)] + [self(q, False, ored) for q in p[1:]]
    def on_notMatches(self, p, inv):
        if inv:
            return [typedThing('Regex', trueOnError=ored)] + [self(q, False, ored) for q in p[1:]]
        return [typedThing(p[0], trueOnError=ored)] + [self(q, False, ored) for q in p[1:]]

    def on_Or(self, p, inv, ored):
        if inv:
            return [ORED_AND] + [self(q, True, True) for q in p[1:]]
        return [p[0]] + [self(q, False, False) for q in p[1:]]
    def on_And(self, p, inv, ored):
        if inv:
            return [OR] + [self(q, True, ored) for q in p[1:]]
        return [p[0]] + [self(q, False, ored) for q in p[1:]]
    def on_BoolVal(self, p, inv, ored):
        if inv and ored:
            return [ORED_NOT, self(p[1], False)]
        if inv:
            return [NOT, self(p[1], False)]
        return [p[0], self(p[1], False)]
        

def on_Boolean_Gen(true, false):
    def on_Boolean(self, p, inv, ored):
        if (inv and p[1] != false) or (not inv) and p[1] == false:
            return (p[0], false)
        return (p[0], true)
    return on_Boolean


class AST(object):
    def __init__(self, ast, sink=None):
        self.ast = ast
        if sink:
            self.sink = sink
        else:
            self.sink = self
    def prod(self, thing):
        return thing[0]
    def abbr(self, prodURI): 
        return prodURI.split('#').pop()
    def run(self):
        self.productions = []
        stack = [[self.ast, 0]]
        while stack:
            if not isinstance(stack[-1][0][1], (tuple, list)):
                a = self.onToken(stack[-1][0][0], stack[-1][0][1])
                if a:
                    return a
                stack.pop()
            elif stack[-1][1] >= len(stack[-1][0]):
                k = self.onFinish()
                stack.pop()
            else:
                k = stack[-1][1]
                stack[-1][1] = k + 1
                if k == 0:
                    self.onStart(stack[-1][0][0])
                else:
                    stack.append([stack[-1][0][k], 0])
        return k
                
        

    def onStart(self, prod):
        if verbose():
            if callable(prod):
                print (' ' * len(self.productions)) + prod()
            else:
                print (' ' * len(self.productions)) + `prod`
        #if callable(prod):
        #    prod = prod()
        self.productions.append([prod])

    def onFinish(self):
      k = self.productions.pop()
      prodName = self.abbr(k[0])
      prod = self.sink.prod(k)
      if self.productions:
          self.productions[-1].append(prod)
      if verbose(): print (' ' * len(self.productions)) + '/' + prodName + ': ' + `prod`
      return prod

    def onToken(self, prod, tok):
      k = self.sink.prod((prod, tok))
      try:
          self.productions[-1].append(k)
      except IndexError:
          return k
      if verbose(): print (' ' * len(self.productions)) + `(prod, tok)`


class productionHandler(object):
    def prod(self, production):
        if hasattr(self, 'on_' + abbr(production[0])):
            try:
                return getattr(self, 'on_' + abbr(production[0]))(production)
            except:
                print production
                raise
        if True: # len(production) > 1:
            raise RuntimeError("why don't you define a %s function, to call on %s?" % ('on_' + abbr(production[0]), `production`))
        return production


class TripleHolder(tuple):
    def __new__(cls, *args, **keywords):
        self = tuple.__new__(cls, *args, **keywords)
        if len(self) != 3:
            raise TypeError
        return self

class FilterExpr(productionHandler):
    def __init__(self, store, parent):
        self.store = store
        self.parent = parent
        self.bnode = parent.new_bnode
        self.string = store.newSymbol('http://www.w3.org/2000/10/swap/string')
        self.anything = self.parent.sparql
        self.math = self.parent.math
        self.log = self.parent.store.newSymbol('http://www.w3.org/2000/10/swap/log')

    def on_function(self, p):
        args = []
        extra = []
        keepGoing = getTrueOnError(p[0])
        rawArgs = p[2:]
        for rawArg in rawArgs:
            if not isinstance(rawArg, tuple):
                return ['Error']
            extra.extend(getExtra(rawArg))
            args.append(tuple(rawArg))
        if p[1] not in knownFunctions:
            raise NotImplementedError('''I don't support the ``%s'' function''' % p[1])
        try:
            node, triples = knownFunctions[p[1]](self, keepGoing, *args)
            return andExtra(node, triples + extra)
        except TypeError:
#            raise
            return ['Error']

    def typeConvert(self, keepGoing, uri, val):
        retVal = self.bnode()
        return (retVal, [makeSafeVal(retVal, (('List', [val[1], self.store.newSymbol(uri)]), ('symbol', self.anything['dtLit']), retVal), safeVersion=keepGoing)])

    def on_BoolVal(self, p):
        extra = getExtra(p[1])
        val = tuple(p[1])
        return [makeTriple(val, ('symbol', self.anything['truthValue']), ('symbol', self.parent.true), safeVersion=getTrueOnError(p[0]))] + extra
    def on_Not(self, p):
        extra = getExtra(p[1])
        val = tuple(p[1])
        if val == ('Error',):
            return ['Error']
        return [makeTriple(val, ('symbol', self.anything['truthValue']), ('symbol', self.parent.false), safeVersion=getTrueOnError(p[0]))] + extra

    def on_And(self, p):
        vals = []
        succeedAnyway = getTrueOnError(p[0])
        things = p[1:]
        for thing in things:
            if thing == ['Error']:
                if succeedAnyway:
                    continue
                return ['Error']
            vals.extend(thing)
        return vals

    def on_Or(self, p):
        p = p[1:]
        returns = []
        for val in p:
            if val != ['Error']:
                returns.extend(self.parent.on_GroupGraphPattern([None, None, val, None], True))
        return [('union', returns)]

    def on_Regex(self, p):
        if str(p[3][1]):
            raise NotImplementedError('I don\'t know how to deal with flags. The flag is: %r' % p[3][1])
        extra = getExtra(p[1]) + getExtra(p[2])
        string = tuple(p[1])
        regex = tuple(p[2])
        return [makeTriple(string, ('symbol', self.string['matches']), regex, safeVersion=getTrueOnError(p[0]))] + extra

    def compare(self, p, op):
        if not isinstance(p[1], tuple) or not isinstance(p[2], tuple):
            return ['Error']
        extra = getExtra(p[1]) + getExtra(p[2])
        op1 = tuple(p[1])
        op2 = tuple(p[2])
        return [makeTriple(op1, ('symbol', op), op2, safeVersion=getTrueOnError(p[0]))] + extra

    def on_less(self, p):
        return self.compare(p, self.anything['lessThan'])
    def on_notLess(self, p):
        return self.compare(p, self.anything['notLessThan'])
    def on_equal(self, p):
        return self.compare(p, self.anything['equals'])
    def on_notEqual(self, p):
        return self.compare(p, self.anything['notEquals'])
    def on_greater(self, p):
        return self.compare(p, self.anything['greaterThan'])
    def on_notGreater(self, p):
        return self.compare(p, self.anything['notGreaterThan'])

    def arithmetic(self, p, op):
        if not isinstance(p[1], tuple) or not isinstance(p[2], tuple):
            return ['Error']
        extra = getExtra(p[1]) + getExtra(p[2])
        op1 = tuple(p[1])
        op2 = tuple(p[2])
        retVal = self.bnode()
        triple = makeSafeVal(retVal, (('List', [op1[1], op2[1]]), ('symbol', op), retVal), safeVersion=getTrueOnError(p[0]))
        return andExtra(retVal, [triple] + extra)
    def on_subtract(self, p):
        return self.arithmetic(p, self.math['difference'])
    def on_add(self, p):
        return self.arithmetic(p, self.math['sum'])
    def on_multiply(self, p):
        return self.arithmetic(p, self.math['product'])
    def on_divide(self, p):
        return self.arithmetic(p, self.math['quotient'])

    def on_Var(self, p):
        return p
    def on_symbol(self, p):
        return p
    def on_Literal(self, p):
        return p
    def on_Boolean(self, p):
        return p
    def on_Number(self, p):
        return p
    def on_String(self, p):
        return p
    def on_funcName(self, p):
        return p[1]

    def on_notBound(self, var):
        var = ('Literal', self.store.newLiteral(var[1][1].uriref()))
        return [makeTriple(self.bnode(), ('symbol', self.parent.sparql['notBound']), var)]
    def on_Bound(self, var):
        var = ('Literal', self.store.newLiteral(var[1][1].uriref()))
        return [makeTriple(self.bnode(), ('symbol', self.parent.sparql['bound']), var)]

    def on_isURI(self, p):
        return [makeTriple(p[1], ('symbol', self.log['rawType']), ('symbol', self.parent.store.Other), safeVersion=getTrueOnError(p[0]))]
    def on_isNotURI(self, p):
        k = self.bnode()
        return [makeTriple(p[1], ('symbol', self.log['rawType']), k, safeVersion=getTrueOnError(p[0])),
                makeTriple(k, ('symbol', self.log['notEqualTo']), ('symbol', self.parent.store.Other))]
    def on_lang(self, p):
        if not isinstance(p[1], tuple):
            return ['Error']
        extra = getExtra(p[1])
        op1 = tuple(p[1])
        retVal = self.bnode()
        meaningLess = self.bnode()
        triple = makeSafeVal(retVal, (('List', [meaningLess[1], retVal[1]]), ('symbol', self.anything['langLit']), p[1]), safeVersion=getTrueOnError(p[0]))
        return andExtra(retVal, [triple] + extra)
    def on_datatype(self, p):
        if not isinstance(p[1], tuple):
            return ['Error']
        extra = getExtra(p[1])
        op1 = tuple(p[1])
        retVal = self.bnode()
        meaningLess = self.bnode()
        triple = makeSafeVal(retVal, (('List', [meaningLess[1], retVal[1]]), ('symbol', self.anything['dtLit']), p[1]), safeVersion=getTrueOnError(p[0]))
        return andExtra(retVal, [triple] + extra)


class FromSparql(productionHandler):
    def __init__(self, store, formula=None, ve=0, why=None):
        verbose(ve)
        self.store = store
        if formula is None:
            self.formula = store.newFormula()
        else:
            self.formula = formula
        self.prefixes = {}
        self.vars = {}
        self.base = 'http://yosi.us/sparql#'
        self.sparql = store.newSymbol(SPARQL_NS)
        self.xsd = store.newSymbol('http://www.w3.org/2001/XMLSchema')
        self.math = store.newSymbol('http://www.w3.org/2000/10/swap/math')
        self.numTypes = Set([self.xsd[k] for k in ['unsignedShort', 'short', 'nonPositiveInteger', 'decimal', 'unsignedInt', 'long', 'nonNegativeInteger', 'int', 'unsignedByte', 'positiveInteger', 'integer', 'byte', 'negativeInteger', 'unsignedLong']])
        self.true = store.newLiteral('true', dt=self.xsd['boolean'])
        self.false = store.newLiteral('false', dt=self.xsd['boolean'])
        self.anonymous_counter = 0
        self.uribase = uripath.base()
        self.dataSets = None
        NotNot.on_Boolean = on_Boolean_Gen(self.true, self.false)
        self._reason = why      # Why the parser w
        _reason2 = None # Why these triples
        if diag.tracking: _reason2 = BecauseOfData(store.newSymbol(self.base), because=self._reason)
        reason2(_reason2)
        self.anNodes = {}

    def anonymize(self, formula, uri = None):
        if uri is not None:
            if isinstance(uri, TripleHolder):
                f = formula.newFormula()
                f.add(*[self.anonymize(formula, k) for k in uri])
                return f.close()
            if isinstance(uri, list):
                return formula.newList([self.anonymize(formula, k) for k in uri])
            if isinstance(uri, Formula):
                return uri.close()
            if isinstance(uri, Term):
                return uri
            try:
                if uri in self.anNodes:
                    return self.anNodes[uri]
            except:
                print uri
                print 'uri = ', uri
                raise
            self.anNodes[uri] = formula.newBlankNode(why=reason2())
            return self.anNodes[uri]
        return formula.newBlankNode(why=reason2())

    def new_bnode(self):
        self.anonymous_counter += 1
        return ('anonymous', '_:%s' % str(self.anonymous_counter))

    def absolutize(self, uri):
        return uripath.join(self.uribase, uri)

    def on_Query(self, p):
        return self.formula

    def on_BaseDecl(self, p):
        self.uribase = p[2][1][1:-1]

    def makePatterns(self, f, node, patterns):
        sparql = self.sparql
        knowledge_base_f = f.newFormula()
        if not self.dataSets:
            knowledge_base = f.newBlankNode()
            f.add(self.uribase, sparql['data'], knowledge_base, why=reason2())
        else:
            knowledge_base = knowledge_base_f.newBlankNode(why=reason2())
            sources = self.store.nil
            #raise RuntimeError(self.dataSets)
            for uri in self.dataSets:
                stuff = knowledge_base_f.newBlankNode(why=reason2())
                uri2 = self.anonymize(knowledge_base_f,uri[1])
                knowledge_base_f.add(uri2, self.store.semantics, stuff, why=reason2())
                sources = sources.prepend(stuff)
            knowledge_base_f.add(sources, self.store.newSymbol('http://www.w3.org/2000/10/swap/log#conjunction'), knowledge_base)
        for pattern in patterns:
            tail = f.newFormula()
            tail.loadFormulaWithSubstitution(pattern[1], why=reason2())
            tail.loadFormulaWithSubstitution(knowledge_base_f, why=reason2())
            includedStuff = pattern[4]
            notIncludedStuff = pattern[5]

            for nodeName, graphIntersection in includedStuff.iteritems():
                if not graphIntersection: continue
                graph = f.newFormula()
                for subGraph in graphIntersection:
                    graph.loadFormulaWithSubstitution(subGraph, why=reason2())
                graph = graph.close()
                if nodeName is not None:
                    nameNode = self.anonymize(tail, nodeName[1])
                    semantics = tail.newBlankNode(why=reason2())
                    tail.add(nameNode, self.store.semantics, semantics, why=reason2())
                else:
                    semantics = knowledge_base
                tail.add(semantics, self.store.includes, graph, why=reason2())
            includedVars = Set(self.vars.values())
            excludedVars = includedVars.difference(tail.occurringIn(includedVars))

            for nodeName, graphIntersection in notIncludedStuff.iteritems():
                if not graphIntersection: continue
##                graph = f.newFormula()
                for subGraph in graphIntersection:
##                    graph.loadFormulaWithSubstitution(subGraph)
##                graph = graph.close()
                    if nodeName is not None:
                        nameNode = self.anonymize(tail, nodeName[1])
                        semantics = tail.newBlankNode(why=reason2())
                        tail.add(nameNode, self.store.semantics, semantics, why=reason2())
                    else:
                        semantics = knowledge_base
                    excludedMap = {}
                    bNodedSubGraph = subGraph.newFormula()
                    for v in excludedVars:
                        excludedMap[v] = bNodedSubGraph.newBlankNode()
                    bNodedSubGraph.loadFormulaWithSubstitution(subGraph, excludedMap)
                    tail.add(semantics, self.store.smartNotIncludes, bNodedSubGraph.close(), why=reason2())

            
            f.add(node, sparql['where'], tail.close(), why=reason2())
##            for parent in pattern[2]:
##                f.add(pattern[1], sparql['andNot'], parent)

    def on_SelectQuery(self, p):
        sparql = self.sparql
        store = self.store
        f = self.formula
##        for v in self.vars:
##            f.declareUniversal(v)
        q = f.newBlankNode()
        f.add(q, store.type, sparql['SelectQuery'], why=reason2())
        variable_results = store.newFormula()
        for v in p[3][1]:
#            variable_results.add(v, store.type, sparql['Binding'])
            variable_results.add(v, sparql['bound'], abbr(v.uriref()), why=reason2())
        f.add(q, sparql['select'], variable_results.close(), why=reason2())

        if p[2]:
            f.add(q, store.type, sparql['Distinct'])

        self.makePatterns(f, q, p[5])
        f3 = RulesMaker(self.sparql).implications(q, f, variable_results)
        for triple in f3.statementsMatching(pred=sparql['implies']):
            f4 = f3.newFormula()
            f4.add(triple.object(), store.type, sparql['Result'], why=reason2())
            f.add(triple.subject(), store.implies, f4.close(), why=reason2())
        #TODO: I'm missing sorting and datasets
        if p[6] and p[6] != (None, None, None):
            raise NotImplementedError('Cwm does not support output modifiers yet')
            sort, limit, offset = p[6]
            if sort:
                l = self.store.newList(sort)
                f.add(q, sparql['sort'], l)
            if limit:
                f.add(q, sparql['limit'], limit)
            if offset:
                f.add(q, sparql['offset'], offset)
#            raise RuntimeError(`p[6]`)
        return None

    def on_ConstructQuery(self, p):
        sparql = self.sparql
        store = self.store
        f = self.formula
##        for v in self.vars:
##            f.declareUniversal(v)
        q = f.newBlankNode()
        f.add(q, store.type, sparql['ConstructQuery'])
        f.add(q, sparql['construct'], p[2])
        knowledge_base = f.newFormula()
        self.makePatterns(f, q, p[4])
        f3 = RulesMaker(self.sparql).implications(q, f, p[2])
        for triple in f3.statementsMatching(pred=sparql['implies']):
            f.add(triple.subject(), store.implies, triple.object())
        return None

    def on_AskQuery(self, p):
        sparql = self.sparql
        store = self.store
        f = self.formula
##        for v in self.vars:
##            f.declareUniversal(v)
        q = f.newBlankNode()
        f.add(q, store.type, sparql['AskQuery'])
        only_result = store.newFormula()
        only_result.add(q, store.type, sparql['Success'])
        only_result = only_result.close()
        self.makePatterns(f, q, p[3])
        f3 = RulesMaker(self.sparql).implications(q, f, only_result)
        for triple in f3.statementsMatching(pred=sparql['implies']):
            f.add(triple.subject(), store.implies, only_result)
        return None

    def on_WhereClause(self, p):
        stuff2 = p[2]
        stuff = []

#        raise RuntimeError(`p`)
        for k in stuff2:
            append = True
            positiveTriples = None
            freeVariables = None
            included = k[5]+k[3]+{None: k[1]}
            notIncluded =  k[6]
##            print '+++++++++++++'
##            print 'included=', included
##            print 'notIncluded=', notIncluded
            for pred, obj in k[4]:
                if positiveTriples is None:
                    positiveTriples = self.store.newFormula()
                    for formSet in included.values():
                        for form in formSet:
                            positiveTriples.loadFormulaWithSubstitution(form)
                    positiveTriples = positiveTriples.close()
                    freeVariables = Set([x.uriref() for x in positiveTriples.freeVariables()])
                if pred is self.sparql['bound']:
                    variable = unicode(obj)
                    if variable not in freeVariables:
                        append = False
                elif pred is self.sparql['notBound']:  ##@@@ This is broken!!
                    variable = unicode(obj)
                    if variable in freeVariables:
                        append = False
            if append:
                stuff.append((k[0], k[2], k[3], k[4], included, notIncluded))
##
##        Formula.doesNodeAppear = realNodeAppear
#        raise RuntimeError(stuff)
        return stuff

    def on_SolutionModifier(self, p):
        if len(p) == 1:
            return None
        return tuple(p[1:])

    def on__QOrderClause_E_Opt(self, p):
        if len(p) == 1:
            return None
        return p[1]

    def on__QLimitClause_E_Opt(self, p):
        if len(p) == 1:
            return None
        return p[1]

    def on__QOffsetClause_E_Opt(self, p):
        if len(p) == 1:
            return None
        return p[1]

    def on__QBaseDecl_E_Opt(self, p):
        return None

    def on_PrefixDecl(self, p):
        self.prefixes[p[2][1][:-1]] = self.absolutize(p[3][1][1:-1])
        self.store.bind(p[2][1][:-1],self.absolutize(p[3][1][1:-1]))
        return None

    def on__QDISTINCT_E_Opt(self, p):
        if len(p) == 1:
            return None
        return None
        raise NotImplementedError(`p`)

    def on_Var(self, p):
        uri = self.base + p[1][1][1:]
        if uri not in self.vars:
            self.vars[uri] = self.formula.newUniversal(uri)
##        self.vars.add(var)
        return ('Var', self.vars[uri])

    def on__QVar_E_Plus(self, p):
        if len(p) == 1:
            return []
        return p[2] + [p[1]]

    def on__O_QVar_E_Plus_Or__QTIMES_E__C(self, p):
        if len(p) == 3:
            varList = [x[1] for x in p[2] + [p[1]]]
        else:
            class ___(object):
                def __iter__(s):
                    return iter(self.vars.values())
            varList = ___()
        return ('SelectVars', varList)

    def on__QDatasetClause_E_Star(self, p):
        return None

    def on_VarOrTerm(self, p):
        return p[1]

    def on_QName(self, p):
        qn = p[1][1].split(':')
        if len(qn) != 2:
            raise RuntimeError
        return ('QuotedIRIref', '<' + self.prefixes[qn[0]] + qn[1] + '>')

    def on_IRIref(self, p):
        return ('symbol', self.store.newSymbol(self.absolutize(p[1][1][1:-1])))

    def on_VarOrBlankNodeOrIRIref(self, p):
        return p[1]

    def on_String(self, p):
        return ('str', unEscape(p[1][1]))

    def on_Verb(self, p):
        if abbr(p[1][0]) == 'IT_a':
            return ('symbol', self.store.type)
        return p[1]

    def on__Q_O_QLANGTAG_E__Or__QDTYPE_E____QIRIref_E__C_E_Opt(self, p):
        if len(p) == 1:
            return (None, None)
        return p[1]

    def on_RDFLiteral(self, p):
        return ('Literal', self.store.newLiteral(p[1][1], dt=p[2][0], lang=p[2][1]))

    def on_NumericLiteral(self, p):
        if abbr(p[1][0]) == 'INTEGER':
            return ('Literal', self.store.newLiteral(`int(p[1][1])`, dt=self.xsd['integer'], lang=None))
        if abbr(p[1][0]) == 'FLOATING_POINT':
            return ('Literal', self.store.newLiteral(`float(p[1][1])`, dt=self.xsd['double'], lang=None))
        raise RuntimeError(`p`)

    def on_RDFTerm(self, p):
        return p[1]

    def on_GraphTerm(self, p):
        return p[1]

    def on_Object(self, p):
        if p[1][0] != 'andExtra':
            return ('andExtra', p[1], [])
        return p[1]

    def on__Q_O_QCOMMA_E____QObjectList_E__C_E_Opt(self, p):
        if len(p) == 1:
            return ('andExtra', [], [])
        return p[1]

    def on_ObjectList(self, p):
        extras = p[2][2] + p[1][2]
        objects = p[2][1] + [p[1][1]]
        return ('andExtra', ('objectList', [k for k in objects]), extras)

    def on__Q_O_QSEMI_E____QPropertyList_E__C_E_Opt(self, p):
        if len(p) == 1:
            return ('andExtra', ('predicateList', []), [])
        return p[1]

    def on_PropertyListNotEmpty(self, p):
        extra = p[2][2] + p[3][2]
        pred = (p[1], p[2][1])
        preds = p[3][1][1] + [pred]
        return ('andExtra', ('predicateList', [k for k in preds]), extra)

    def on_Triples1(self, p):
        if abbr(p[1][0]) == 'GT_LBRACKET':
            return p[2]
        if abbr(p[1][0]) == 'GT_LPAREN':
            return p[2]
        extra = p[2][2]
        return [('Triple', (p[1], p[2][1]))] + extra

    def on_Triples2(self, p):
        if len(p) == 4:
            predList = ('predicateList', p[1][1][1] + p[3][1][1])
            extra = p[1][2] + p[3][2]
        else:
            predList = p[2][1]
            extra = p[2][2]
        return [('Triple', (self.new_bnode(), predList))] + extra


    def on_Triples3(self, p):
        store = self.store
        if len(p) == 3:
            return [('Triple', (('symbol', store.nil), p[2][1]))] + p[2][2]
        extra = p[1][2] + p[2][2] + p[4][2]
        nodes = [p[1][1]] + p[2][1]
        pred = p[4][1]
        realPred = pred[1]
        if realPred == []:
            realPred.append((('symbol', self.sparql['lamePred']), ('objectList', [('symbol', self.sparql['LameObject'])])))
        List = ('List', [k[1] for k in nodes])
        return [('Triple', (List, pred))] + extra
    

    def on_GraphPatternListTail(self, p):
        if len(p) == 1:
            return []
        return p[1]

    def  on__O_QTriples1_E____QGraphPatternListTail_E__Or__QGraphPatternNotTriples_E____QGraphPatternNotTriplesTail_E__C(self, p):
        return p[2] + p[1]

    def on__Q_O_QTriples1_E____QGraphPatternListTail_E__Or__QGraphPatternNotTriples_E____QGraphPatternNotTriplesTail_E__C_E_Opt(self, p):
        if len(p) == 1:
            return []
        return p[1]

    def on_GraphPatternList(self, p):
        if len(p) == 1:
            return []
        if len(p) == 2:
            return p[1]
        return p[1] + p[2]

    def on__O_QDot_E____QGraphPatternList_E__C(self, p):
        return p[2]

    def on__Q_O_QDot_E____QGraphPatternList_E__C_E_Opt(self, p):
        if len(p) == 1:
            return []
        return p[1]

    def on_GroupGraphPattern(self, p, fromFilter = False):
        store = self.store
        triples = p[2]
        options = []
        alternates = []
        parents = multimap()
        bounds = []
        subGraphs = []
        f = self.store.newFormula()
        for triple in triples:
            if triple == 'Error':
                return []
            try:
                if triple[0] == 'union':
                    alternates.append([k[1:] for k in triple[1]])
                    continue
                if triple[0] == 'SubGraph':
                    subGraphs.append(triple[1:])
                    continue
                rest1 = triple[1]
                subject = rest1[0]
                predicateList = rest1[1][1]
                for rest2 in predicateList:
                    predicate = rest2[0]
                    objectList = rest2[1][1]
                    
                    for object in objectList:
                        try:
                            subj = self.anonymize(f, subject[1])
                            pred = self.anonymize(f, predicate[1])
                            if pred is self.sparql['OPTIONAL']:
                                options.append(object)
                            else:
                                obj = self.anonymize(f, object[1])
                                if pred is self.sparql['bound'] or pred is self.sparql['notBound']:
                                    bounds.append((pred, obj))
                                    
                                    #print alternates, object[1], isinstance(object[1], Term), self.anonymize(f, object[1])
##                                elif isinstance(obj, Formula):
##                                    options.extend(object[2])
##                                    f.add(subj, pred, obj)
                                else:
                                    f.add(subj, pred, obj, why=reason2())
                        except:
                            print '================'
                            print 'subject= ', subject
                            print 'predicate= ', predicate
                            print 'pred= ', pred is self.sparql['OPTIONAL'], id(pred), id(self.sparql['OPTIONAL'])
                            print 'object= ', object
                            raise
            except:
                print 'triples=',triples
                print 'triple=',triple
                raise

        f = f.close()
        if fromFilter:
            retVal = [('formula', self.store.newFormula(), f, parents, bounds, multimap(), multimap())]
        else:
            retVal = [('formula', f, self.store.newFormula(), parents, bounds, multimap(), multimap())]
        for nodeName, subGraphList in subGraphs:
            oldRetVal = retVal
            retVal = []
            for _, f, filters, p, b, i, nI in oldRetVal:
                node = self.anonymize(f, nodeName[1])
                for __, subF, filters2, p2, b2, i2, nI2 in subGraphList:
                    #@@@@@  What do I do with b2?
                    newF = f.newFormula()
                    newF.loadFormulaWithSubstitution(f)
                    newFilters =  f.newFormula()
                    newFilters.loadFormulaWithSubstitution(filters)
                    newFilters.loadFormulaWithSubstitution(filters2)
                    i2.translate(None, nodeName)
                    nI2.translate(None, nodeName)
                    p2.translate(None, nodeName)
                    retVal.append(('formula', newF.close(), newFilters.close(), p+p2, b+b2, i + i2 + {nodeName: subF}, nI + nI2))
        for alternate in alternates:
            oldRetVal = retVal
            retVal = []
            for formula1, filters1, parents1, bounds1, include1, notInclude1 in alternate:
                for ss, formula2, filters2, parents2, bounds2, include2, notInclude2 in oldRetVal:
                    f = self.store.newFormula()
                    if formula1:
                        f.loadFormulaWithSubstitution(formula1)
                    f.loadFormulaWithSubstitution(formula2)
                    
                    newFilters =  f.newFormula()
                    newFilters.loadFormulaWithSubstitution(filters1)
                    newFilters.loadFormulaWithSubstitution(filters2)
                    retVal.append(('formula', f.close(), newFilters.close(), parents1 + parents2, bounds1 + bounds2, include1 + include2, notInclude1 + notInclude2))
        for alternate in options:
            oldRetVal = retVal
            retVal = []
            for ss, formula1, filters1, parents1, bounds1, i1, nI1 in alternate:
                for ss, formula2, filters2, parents2, bounds2, i2, nI2 in oldRetVal:
                    f1 = self.store.newFormula()
                    f1.loadFormulaWithSubstitution(formula1, why=reason2())
                    f1.loadFormulaWithSubstitution(formula2, why=reason2())
                    f1 = f1.close()
                    f2 = self.store.newFormula()
                    f2.loadFormulaWithSubstitution(formula2, why=reason2())

                    f3 = formula1.newFormula()
                    f3.loadFormulaWithSubstitution(formula1, why=reason2())
                    for document in i1:
                        semantics = f3.newBlankNode()
                        f3.add(document[1], self.store.semantics, semantics)
                        totalFormula = f3.newFormula()
                        for f4 in i1[document]:
                            totalFormula.loadFormulaWithSubstitution(f4, why=reason2())
                        f3.add(semantics, self.store.includes, totalFormula.close())
                    f3.loadFormulaWithSubstitution(filters1, why=reason2())
                    f3 = f3.close()
                    newFilters1 =  f1.newFormula()
                    newFilters1.loadFormulaWithSubstitution(filters1, why=reason2())
                    newFilters1.loadFormulaWithSubstitution(filters2, why=reason2())

                    newFilters2 =  f2.newFormula()
                    newFilters2.loadFormulaWithSubstitution(filters2, why=reason2())                    
                    
                    retVal.append(('formula', formula2.close(), newFilters1.close(), parents1 + parents2+{None:formula1}, bounds1 + bounds2, i1+i2, nI1+nI2))
                    retVal.append(('formula', formula2.close(), newFilters2.close(), parents2, bounds1 + bounds2, i2, nI2+{None:f3}))
        return retVal
##        
##        if len(p) == 2:
##            p.append([])
##        if p[1][0][0] == 'Triple':
##            p[2] = p[1][1:] + p[2]
##            p[1] = p[1][0]
##        if p[1][0] == 'Triple':
##            
##            
##        elif p[1][0][0] == 'formula':
##            if p[2]:
##                raise RuntimeError(`p`)
##            graphs = p[1]
##            return graphs
##        else:
##            raise RuntimeError(`p`)

    def on__QPropertyListNotEmpty_E_Opt(self, p):
        if len(p) == 1:
            return ('andExtra', ('predicateList', []), [])
        return p[1]

    def on_PropertyList(self, p):
        return p[1]

    def on_NamelessBlank(self, p):
        return self.new_bnode()

    def on_BlankNode(self, p):
        return ('anonymous', p[1][1])

    def on_BlankNodePropertyList(self, p):
        extra = p[2][2]
        preds = p[2][1]
        anon = self.new_bnode()
        extra.append(('Triple', (anon, preds)))
        return  ('andExtra', anon, extra)

    def on_TriplesNode(self, p):
        return p[1]

    def on__O_QSEMI_E____QPropertyList_E__C(self, p):
        return p[2]


    def on_GraphNode(self, p):
        if p[1][0] != 'andExtra':
            return ('andExtra', p[1], [])
        return p[1]

    def on__QGraphNode_E_Plus(self, p):
        return self.on__QGraphNode_E_Star(p)


    def on__QGraphNode_E_Star(self, p):
        if len(p) == 1:
            return ('andExtra', [], [])
        nodes = [p[1][1]] + p[2][1]
        extra = p[1][2] + p[2][2]
        return ('andExtra', nodes, extra)

    def on_Collection(self, p):
        extra = p[2][2]
        nodes = p[2][1]
        List = ('List', [k[1] for k in nodes])
        return ('andExtra', List, extra)

    def on_GraphPatternNotTriplesList(self, p):
        return p[1] + p[2]
    
### End Triples Stuff
#GRAPH
    def on_GraphGraphPattern(self, p):
        return [('SubGraph', p[2], p[3])]
##        semantics = self.new_bnode()
##        return [makeTriple(p[2], ('symbol', self.store.semantics), semantics),
##                ('SubGraph', semantics, p[3])
###                makeTripleObjList(semantics, ('symbol', self.store.includes), p[3])
##                ]
#OPTIONAL
    def on_GraphPatternNotTriples(self, p):
        return p[1]

    def on_GraphPatternNotTriplesTail(self, p):
        if len(p) == 1:
            return []
        return p[1]

    def on__O_QDot_E_Opt___QGraphPatternList_E__C(self, p):
        return p[2]

    def on_OptionalGraphPattern(self, p):
        return [makeTriple(self.new_bnode(), ('symbol', self.sparql['OPTIONAL']), p[2])]
#UNION
    def on__O_QUNION_E____QGroupGraphPattern_E__C(self, p):
        return p[2]

    def on__Q_O_QUNION_E____QGroupGraphPattern_E__C_E_Star(self, p):
        if len(p) == 1:
            return []
        return p[1] + p[2]

    def on_GroupOrUnionGraphPattern(self, p):
        return [('union', p[1] + p[2])]

#FILTER
    def on_PrimaryExpression(self, p):
        return p[1]

    def on_UnaryExpression(self, p):
        if len(p) == 2:
##            if getType(p[1][0]) != 'boolean':
##                return (typedThing('BoolVal', 'boolean'), p[1])
            return p[1]
        if abbr(p[1][0]) == 'GT_NOT':
            return (typedThing('Not', 'boolean'), p[2])
        raise RuntimeError(`p`)

    def on__Q_O_QTIMES_E____QUnaryExpression_E__Or__QDIVIDE_E____QUnaryExpression_E__C_E_Star(self, p):
        if len(p) == 1:
            return []
        if not p[2]:
            return p[1]
        return (p[1][0], (p[2][0], p[1][1], p[2][1]))

    def on_MultiplicativeExpression(self, p):
        if p[2] == []:
            return p[1]
        return (p[2][0], p[1], p[2][1])

    def on__Q_O_QPLUS_E____QMultiplicativeExpression_E__Or__QMINUS_E____QMultiplicativeExpression_E__C_E_Star(self, p):
        if len(p) == 1:
            return []
        if not p[2]:
            return p[1]
        return (p[1][0], (p[2][0], p[1][1], p[2][1]))

    def on_AdditiveExpression(self, p):
        if p[2] == []:
            return p[1]
        return (p[2][0], p[1], p[2][1])

    def on_NumericExpression(self, p):
        return p[1]

    def on__Q_O_QEQUAL_E____QNumericExpression_E__Or__QNEQUAL_E____QNumericExpression_E__Or__QLT_E____QNumericExpression_E__Or__QGT_E____QNumericExpression_E__Or__QLE_E____QNumericExpression_E__Or__QGE_E____QNumericExpression_E__C_E_Opt(self, p):
        if len(p) == 1:
            return None
        return p[1]


    def on_RelationalExpression(self, p):
        if p[2] is None:
            return p[1]
##        if p[2][0] != 'less':
##            raise RuntimeError(p[2], getType(p[2][1][0]))
        if p[2][0] == 'equal':
            t1, t2 = getType(p[1][0]), getType(p[2][1][0])
            if t1 == 'boolean' or t2 == 'boolean':
                return (OR, (AND, p[1], p[2][1]), (AND, (NOT, p[1]), (NOT, p[2][1])))
        if p[2][0] == 'notEqual':
            t1, t2 = getType(p[1]), getType(p[2][1])
            if t1 == 'boolean' or t2 == 'boolean':
                return (OR, (AND, (NOT, p[1]), p[2][1]), (AND, p[1], (NOT, p[2][1])))
        return (typedThing(p[2][0], 'boolean'), p[1], p[2][1])

    def on_ValueLogical(self, p):
        return p[1]

    def on__Q_O_QAND_E____QValueLogical_E__C_E_Star(self, p):
        if len(p) == 1:
            return []
        return [p[1]] + p[2]

    def on_ConditionalAndExpression(self, p):
        if p[2]:
            return [AND, p[1]] + p[2]
        return p[1]

    def on__Q_O_QOR_E____QConditionalAndExpression_E__C_E_Star(self, p):
        if len(p) == 1:
            return []
        return [p[1]] + p[2]

    def on_ConditionalOrExpression(self, p):
        if p[2]:
            return [OR, p[1]] + p[2]
        return p[1]

    def on_Expression(self, p):
        return p[1]

    def on_BrackettedExpression(self, p):
        return p[2]

    def on__O_QBrackettedExpression_E__Or__QCallExpression_E__C(self, p):
        """see normalize for an explanation of what we are doing"""
        return normalize(p[1])
    
    def on_Constraint(self, p):
        val = AST(p[2], FilterExpr(self.store, self)).run()
        return [('union', self.on_GroupGraphPattern([None, None, val, None], True))]

#useless
    def on__QPrefixDecl_E_Star(self, p):
        return None
    def on_Prolog(self, p):
        return None
    def on__QWHERE_E_Opt(self, p):
        return None
    def on__O_QSelectQuery_E__Or__QConstructQuery_E__Or__QDescribeQuery_E__Or__QAskQuery_E__C(self, p):
        return None
    def on__QDot_E_Opt(self, p):
        return None

### AutoGenerated

    def on_OffsetClause(self, p):
        return self.on_NumericLiteral(p[1:])[1]

    def on_DescribeQuery(self, p):
        raise RuntimeError(`p`)

    def on__QVarOrIRIref_E_Plus(self, p):
        raise RuntimeError(`p`)

    def on__O_QVarOrIRIref_E_Plus_Or__QTIMES_E__C(self, p):
        raise RuntimeError(`p`)

    def on__QWhereClause_E_Opt(self, p):
        raise RuntimeError(`p`)

    def on_DatasetClause(self, p):
        return None

    def on__O_QDefaultGraphClause_E__Or__QNamedGraphClause_E__C(self, p):
        return None

    def on_DefaultGraphClause(self, p):
        if not self.dataSets:
            self.dataSets = []
        self.dataSets.append(p[1])
        return None

    def on_NamedGraphClause(self, p):
        return None

    def on_SourceSelector(self, p):
        return p[1]

    def on_OrderClause(self, p):
        clauses = [p[3]] + p[4]
        return clauses
        raise RuntimeError(`p`)

    def on__QOrderCondition_E_Plus(self, p):
        if len(p) == 1:
            return []
        return [p[1]] + p[2]
        raise RuntimeError(`p`)

    def on_OrderCondition(self, p):
        def listize(thing):
            if len(thing) == 2 and isinstance(thing[1], Term):
                return thing[1]
            if thing[0] == 'function':
                return self.store.newList([self.store.newSymbol(thing[1][1])] + [listize(x) for x in thing[2:]])
            return self.store.newList([self.store.newLiteral(thing[0])] + [listize(x) for x in thing[1:]])
        return listize(p[1])
        raise RuntimeError(`p`)

    def on__O_QASC_E__Or__QDESC_E__C(self, p):
        return p[1][1]

    def on__O_QASC_E__Or__QDESC_E____QBrackettedExpression_E__C(self, p):
        return p[1:]

    def on__O_QFunctionCall_E__Or__QVar_E__Or__QBrackettedExpression_E__C(self, p):
        return p[1]

    def on_LimitClause(self, p):
        return self.on_NumericLiteral(p[1:])[1]

    def on_ConstructTemplate(self, p):
        return self.on_GroupGraphPattern(p)[0][1]

    def on__QTriples_E_Opt(self, p):
        if len(p) == 1:
            return []
        return p[1]

    def on_Triples(self, p):
        return p[1] + p[2]

    def on__QTriples_E_Opt(self, p):
        if len(p) == 1:
            return []
        return p[1]

    def on__O_QDot_E____QTriples_E_Opt_C(self, p):
        return p[2]

    def on__Q_O_QDot_E____QTriples_E_Opt_C_E_Opt(self, p):
        if len(p) == 1:
            return []
        return p[1]

    def on__O_QCOMMA_E____QObjectList_E__C(self, p):
        #raise RuntimeError(`p`)
        return (p[2][0], p[2][1][1], p[2][2])

    def on_VarOrIRIref(self, p):
        raise RuntimeError(`p`)

    def on__O_QOR_E____QConditionalAndExpression_E__C(self, p):
        return p[2]

    def on__O_QAND_E____QValueLogical_E__C(self, p):
        return (AND, p[2])

    def on__O_QEQUAL_E____QNumericExpression_E__Or__QNEQUAL_E____QNumericExpression_E__Or__QLT_E____QNumericExpression_E__Or__QGT_E____QNumericExpression_E__Or__QLE_E____QNumericExpression_E__Or__QGE_E____QNumericExpression_E__C(self, p):
        op = p[1][1]
        opTable = { '>': 'greater',
                    '<': 'less',
                    '>=': 'notLess',
                    '<=': 'notGreater', '=': 'equal', '!=': 'notEqual'}
        return (opTable[op], p[2])

    def on__O_QPLUS_E____QMultiplicativeExpression_E__Or__QMINUS_E____QMultiplicativeExpression_E__C(self, p):
        return ({'+': typedThing('add', 'number'), '-': typedThing('subtract', 'number')}[p[1][1]], p[2])

    def on__O_QTIMES_E____QUnaryExpression_E__Or__QDIVIDE_E____QUnaryExpression_E__C(self, p):
        return ({'*': typedThing('multiply', 'number'), '/': typedThing('divide', 'number')}[p[1][1]], p[2])

    def on_CallExpression(self, p):
        return p[1]

    def on_BuiltinCallExpression(self, p):
        if len(p) == 2:
            return p[1]
        funcName = abbr(p[1][0])
        if funcName == 'IT_BOUND':
            return (typedThing('Bound', 'boolean', ['variable']), p[3])
        if funcName == 'IT_isURI':
            return (typedThing('isURI', 'boolean'), p[3])
        if funcName == 'IT_STR':
            return (typedThing('String', 'literal', ['literal', 'symbol']), p[3])
        if funcName == 'IT_LANG':
            return (typedThing('lang', 'literal', ['literal']), p[3])
        if funcName == 'IT_DATATYPE':
            return (typedThing('datatype', 'symbol', ['literal']), p[3])
        if funcName == 'IT_isBLANK':
            return (typedThing('isBlank', 'boolean'), p[3])
        if funcName == 'IT_isLITERAL':
            return (typedThing('isLiteral', 'boolean'), p[3])
        raise RuntimeError(`p`)

    def on_RegexExpression(self, p):
        return ('Regex', p[3], p[5], p[6]) 

    def on__O_QCOMMA_E____QExpression_E__C(self, p):
        return p[2]

    def on__Q_O_QCOMMA_E____QExpression_E__C_E_Opt(self, p):
        if len(p) == 1:
            return ('Literal', self.store.newLiteral(''))
        return p[1]

    def on_FunctionCall(self, p):
        return ['function', ("funcName", p[1][1].uriref(),)] + p[2]

    def on_ArgList(self, p):
        return p[2]

    def on__Q_O_QCOMMA_E____QExpression_E__C_E_Star(self, p):
        if len(p) == 1:
            return []
        return [p[1]] + p[2]

    def on__O_QExpression_E____QCOMMA_E____QExpression_E_Star_C(self, p):
        return [p[1]] + p[2]

    def on__Q_O_QExpression_E____QCOMMA_E____QExpression_E_Star_C_E_Opt(self, p):
        if len(p) == 1:
            return []
        return p[1]

    def on_RDFTermOrFunc(self, p):
        if p[1][0] == 'Literal':
            lit = p[1][1]
            if not lit.datatype:
                return (typedThing('String', 'string'), lit)
            if lit.datatype in self.numTypes:
                return (typedThing('Number', 'number'), lit)
            if lit.datatype == self.xsd['boolean']:
                return (typedThing('Boolean', 'boolean'), lit)
        return p[1]

    def on_IRIrefOrFunc(self, p):
        if p[2] == None:
            return p[1]
        return ['function', ("funcName", p[1][1].uriref(),)] + p[2]

    def on__QArgList_E_Opt(self, p):
        if len(p) == 1:
            return None
        return p[1]

    def on__O_QDTYPE_E____QIRIref_E__C(self, p):
        return (p[1][0], p[2][1])

    def on__O_QLANGTAG_E__Or__QDTYPE_E____QIRIref_E__C(self, p):
        if abbr(p[1][0]) == 'LANGTAG':
            return (None, p[1][1][1:])
        if abbr(p[1][0]) == 'GT_DTYPE':
            return (p[1][1], None)
        raise RuntimeError(`p`)
        

    def on_BooleanLiteral(self, p):
        return ('Literal', (p[1][1] == u'true' and self.true or self.false))


class RulesMaker(object):
    def __init__(self, ns):
        self.ns = ns

    def implications(self, query, formula, totalResult):
        retFormula = formula.newFormula()
        for where in formula.each(subj=query, pred=self.ns['where']):
            F = formula.newFormula()
            F.existentials().update(totalResult.existentials())
            bound_vars = self.find_vars(formula.universals(), where)
#            print where, bound_vars
            unbound_vars = formula.universals() - bound_vars
            self.matching_subformula(F, unbound_vars, totalResult)
            retFormula.add(where, self.ns['implies'], F.close(), why=reason2())
        return retFormula

    def find_vars(self, vars, f):
#        print 'find_vars on:', f, vars
        retVal = Set()
        for var in vars:
            if f.contains(subj=var) or f.contains(pred=var) or f.contains(obj=var):
                retVal.add(var)
        for statement in f.statementsMatching(pred=f.store.includes):
            retVal.update(self.find_vars(vars, statement.object()))
#        print 'find_vars found:', retVal, ' on ', f, vars
        return retVal

    def matching_subformula(self, retF, illegals, F):
        def clear(node):
            return node not in illegals and not (isinstance(node, CompoundTerm) and
                                             False in [clear(a) for a in node])
        for triple in F:
            if (clear(triple.subject()) and
                clear(triple.predicate()) and
                clear(triple.object())):
                retF.add(*triple.spo(), **{'why':reason2()})
#            else:
#                print triple, illegals


def unEscape(string):
    if string[:1] == '"':
        delin = '"'
        if string[:3] == '"""':
            real_str = string[3:-3]
            triple = True
        else:
            real_str = string[1:-1]
            triple = False
    else:
        delin = "'"
        if string[:3] == "'''":
            real_str = string[3:-3]
            triple = True
        else:
            real_str = string[1:-1]
            triple = False
    ret = u''
    n = 0
    while n < len(real_str):
        ch = real_str[n]
        if ch == '\r':
            pass
        elif ch == '\\':
            a = real_str[n+1:n+2]
            if a == '':
                raise RuntimeError
            k = 'abfrtvn\\"\''.find(a)
            if k >= 0:
                ret += '\a\b\f\r\t\v\n\\"\''[k]
                n += 1
            elif a == 'u':
                m = real_str[n+2:n+6]
                assert len(m) == 4
                ret += unichr(int(m, 16))
                n += 5
            elif a == 'U':
                m = real_str[n+2:n+10]
                assert len(m) == 8
                ret += unichr(int(m, 16))
                n += 9
            else:
                raise ValueError('Bad Escape')
        else:
            ret += ch
                
        n += 1
        
        
    return ret

def intConvert(self, keepGoing, val):
    return self.typeConvert(keepGoing, 'http://www.w3.org/2001/XMLSchema#integer', val)

knownFunctions['http://www.w3.org/2001/XMLSchema#integer'] = intConvert


def sparqlLookup(uri, server, property):
    pass


