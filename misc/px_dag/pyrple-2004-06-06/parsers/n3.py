#!/usr/bin/python
"""
Notation3 Parser for Pyrple, 2003-10
Derived from: Afon--A Fairly Ostensive Notation3 parser, 2002-07

by Sean B. Palmer

This module parses N3 and outputs quads--comprising (TYPE, VALUE) 
articles--into a sink. It supports the new ?x universally quantified 
variable syntax, as well as => for implication. afon.N3Parser is 
the main parser class, and afon.N3Sink is the abstract sink superclass; 
there are also various tokenization and pre-processing functions, as 
well as a couple of test sinks.

<sbp> in keeping with the cwm/llyn theme, I'm thinking of using afon
<sbp> "a river of N3"
<AaronSw> the river in the valley
<AaronSw> the river to the storage area in the center of the valley

Notation3 references:-

http://www.w3.org/DesignIssues/Notation3
http://www.w3.org/2000/10/swap/Primer
http://www.w3.org/2000/10/swap/Examples

Afon is based on earlier code by Dan Connolly (whose code in particular 
forms the basis of just about every N3 parser in the world to date), and 
by the author:-

* http://www.w3.org/2000/10/swap/notation3.py
* http://www.w3.org/2000/10/swap/rdfn3.g
* http://infomesh.net/2002/eep3/20020703/rdfn3.g.txt
* http://infomesh.net/2002/n3s/n3s.txt

The former two made available under the W3C's software license:-

* http://www.w3.org/2000/10/swap/LICENSE.rdf

The latter two made available under GPL 2.

Thanks to Aaron Swartz and deltab for their reviews and suggestions.
"""

import re, base64
# import uripath # http://www.w3.org/2000/10/swap/uripath.py

import urlparse
import sys as uripath
uripath.join = urlparse.urljoin # yeah. I know, I know...

__license__ = u"\u00A9 Sean B. Palmer 2002, GPL 2."
__version__ = "$Date: 2002-08-05 02:20:55 $"

# Configuration

USE_PATHS = 0 # To use RDFPath, set this to 1

# Tokens

def group(*n): 
    return '(%s)' % '|'.join(n)

NAME = r'[A-Za-z0-9_-]+'
URIREF = r'<[^ >]*>'
EXIVAR = r'_:' + NAME
UNIVAR = r'\?' + NAME
PREFIX = r'(?:[A-Za-z][A-Za-z0-9]*)?:'
QNAME = PREFIX + NAME
LITERALA = r'"[^"\\]*(?:\\.[^"\\]*)*"'
LITERALB = r'"""[^"\\]*(?:(?:\\.|"(?!""))[^"\\]*)*"""'
DECL = r'@[A-Za-z]+'

if USE_PATHS: 
   PATHTERM = r'(?:%s|%s|%s|%s|%s|a)' % \
        (LITERALA, URIREF, QNAME, EXIVAR, UNIVAR)
   PATH = r'%s(?:(?:\.|\^)%s)+' % (PATHTERM, PATHTERM)
   LITERALB += '|' + PATH # if only to get a "hork" out of bijan...

Tokens = group(LITERALB, URIREF, LITERALA, DECL, ':-', QNAME, EXIVAR, 
   PREFIX, NAME, UNIVAR, 'is', 'of', '=>', '=', '{', '}', '\(', '\)', 
   '\[', '\]', ',', ';', '\.')
Token = re.compile(Tokens, re.S)

# # # # # # # # # # # # # # # # #
# 
# Pre-processor and Tokenizer
# 

def preProcess(s): 
    """A simple Notation3 pre-processor; removes "#" comments."""
    s = '\n'.join(s.splitlines())
    return ''.join(re.compile(r'(%s|%s|%s|\n|[ \t]+|[^\s#])|(?:#[^\n]*)' % \
        (URIREF, LITERALB, LITERALA)).findall(s))

def tokenize(s, pre=1): 
    """Notation3 tokenizer. Takes in a string, returns a raw token list."""
    s = s.strip()
    if len(s) == 0: return []
    if pre: s = preProcess(s)
    return Token.findall(s)

#
# # # # # # # # # # # # # # # # #

class N3SyntaxError(SyntaxError):
    def __init__(self, msg, uri, tpos, n3): 
	self.msg = msg
        self.uri = uri
        self.tpos = tpos
        self.n3 = n3

    def __str__(self): 
        return '"%s", <%s> token %s, at (!) in: \n"%s"' % \
                 (self.msg, self.uri, self.tpos, self.n3)

URI, EXI, UNI, LIT = 'URIRef', 'Exivar', 'Univar', 'Lit'
FORMULA = 'FORMULA'

LOG_forSome = (URI, 'http://www.w3.org/2000/10/swap/log#forSome')
LOG_forAll = (URI, 'http://www.w3.org/2000/10/swap/log#forAll')
LOG_implies = (URI, 'http://www.w3.org/2000/10/swap/log#implies')
RDF_type = (URI, 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
DAML_equivalentTo = (URI, 'http://www.daml.org/2001/03/daml+oil#equivalentTo')
DAML_first = (URI, 'http://www.daml.org/2001/03/daml+oil#first')
DAML_rest = (URI, 'http://www.daml.org/2001/03/daml+oil#rest')
DAML_nil = (URI, 'http://www.daml.org/2001/03/daml+oil#nil')

SWAPCOMPAT = 0

DECLmatch = re.compile('^%s$' % DECL).match
PREFIXmatch = re.compile('^%s$' % PREFIX).match
LITERALBmatch = re.compile('^%s$' % LITERALB).match
LITERALAmatch = re.compile('^%s$' % LITERALA).match
URIREFmatch = re.compile('^%s$' % URIREF).match
QNAMEmatch = re.compile('^%s$' % QNAME).match
EXIVARmatch = re.compile('^%s$' % EXIVAR).match
UNIVARmatch = re.compile('^%s$' % UNIVAR).match
if USE_PATHS: PATHmatch = re.compile('^%s$' % PATH).match

class N3Parser: 
    """An N3Document."""

    def __init__(self, sink, baseURI, bindings={}, formulaURI=None): 
        if SWAPCOMPAT: sink = SWAPSink(sink)
        self._sink = sink
        self.tokens = []
        self.pos = 0
        self._baseURI = baseURI
        self._bindings = bindings

        self._bNodes = {}
        self._univars = {}
        self._bNodesList = []
        self._univarsList = []
        self._vars = {}
        self._serial = 0

        if type(formulaURI) is type(''): 
            self.rootFormula = (FORMULA, formulaURI)
        elif formulaURI: self.rootFormula = formulaURI
        else: 
            formulaURI = 'formulae:%s$@' % self._baseURI.replace('$', '$$')
            self.rootFormula = (FORMULA, formulaURI)
        self.parentFormula = self.rootFormula[:]
        self._parent = {self.rootFormula: self.rootFormula}

    def load(self, uri, baseURI=''): 
        if uri: 
            import urllib
            uri = uripath.join(baseURI, uri)
            self._sink.makeComment("Taking input from " + uri)
            self.startDoc()
            self.feed(urllib.urlopen(uri).read())
            self.endDoc()
        else: 
            import sys
            self._sink.makeComment("Taking input from standard input")
            self.startDoc()
            self.feed(sys.stdin.read())
            self.endDoc()

    def feed(self, s): 
        """Feed a string into the parser."""
        self.tokens = tokenize(s)
        self.document()

    def startDoc(self):
        self._sink.startDoc(self.rootFormula, self._baseURI)

    def endDoc(self):
        self._sink.endDoc()

    def syntaxError(self, msg, pos=None, fore=3, aft=5): 
        """Return a customized N3SyntaxError."""
        if pos is None: pos = self.pos
        # You can overslice, but you can't underslice...
        if fore > pos: fro = 0
        else: fro = pos - fore

        # Put in a marker to show where the error was
        self.tokens[pos] = '(!)' + self.tokens[pos]

        tokens = self.tokens[fro:(pos+aft)]
        return N3SyntaxError(msg, self._baseURI, pos, ' '.join(tokens))

    def readTokens(self, n): 
        result = self.tokens[self.pos:(self.pos+n)]
        self.pos += n
        return tuple(result)

    def readToken(self): 
        """Equivalent to self.readTokens(1)[0]."""
        result = self.tokens[self.pos]
        self.pos += 1
        return result

    def document(self): 
        self.scp = self.rootFormula
        while self.pos < len(self.tokens): 
            self.directiveOrStatement()

    def directiveOrStatement(self): 
        """( directive | statement )"""
        if DECLmatch(self.tokens[self.pos]): self.directive()
        else: self.statement()

    def directive(self): 
        """( prefix | forAll | forSome )"""
        t = self.readToken()
        if t == '@prefix': self.prefix()
        elif t == '@forAll': self.quantDecl('forAll')
        elif t == '@forSome': self.quantDecl('forSome')
        else: raise self.syntaxError("Unknown directive.", self.pos-1)

    def prefix(self): 
        prefix, ns, d = self.readTokens(3)
        if not PREFIXmatch(prefix): 
           # we've just read 3 tokens, so the error's at pos-1
           raise self.syntaxError("Bad Prefix", self.pos-1)
        prefix = prefix[:-1] # strip the trailing colon
        ns = ns[1:-1] # strip the angular brackets
        self.bind(prefix, ns)

    def bind(self, prefix, ns): 
        if ns.endswith('#'): ns, sep = ns[:-1], '#'
        else: sep = ''
        ns = uripath.join(self._baseURI, ns) + sep
        self._bindings[prefix] = ns
        self._sink.bind(prefix, (URI, ns))

    def quantDecl(self, q): 
        t = self.readToken()

        if URIREFmatch(t): t = self.uriref(t)
        elif QNAMEmatch(t): t = self.qname(t)
        else: raise "Wrong, wrong, wrong! @@"
        self.refQuant(self.scp, t, q)

        while self.tokens[self.pos] == ',': 
            self.readToken()
            t = self.readToken()
            if URIREFmatch(t): t = self.uriref(t)
            elif QNAMEmatch(t): t = self.qname(t)
            else: raise "Wrong, wrong, wrong! @@"
            self.refQuant(self.scp, t, q)

        assert self.readToken() == '.'

    def statement(self):
        """clause '.'"""
        clause = self.clause()
        if self.readToken() != '.': 
           raise self.syntaxError("Expected a full stop.", self.pos-1)

    def clause(self): 
        """( phrase [ popair ( ';' popair )* ] ) 
           | ( term popair ( ';' popair )* )"""
        t = self.tokens[self.pos]
        if t == '[': 
            phrase = self.phrase()
            if self.tokens[self.pos] not in ('.', '}'): 
                popair = self.popair(phrase)
                while self.tokens[self.pos] == ';': 
                    self.readToken()
                    popair = self.popair(phrase)
        else: 
            term = self.term()
            popair = self.popair(term)
            while self.tokens[self.pos] == ';': 
                self.readToken()
                popair = self.popair(term)

    def phrase(self): 
        """'[' [ ( ( ':-' term ) | ( popair ) ) ( ';' popair )* ] ']'"""
        self.readToken() # we know that it's "["
        subj = self.something("thing", 1)
        if self.tokens[self.pos] == ':-': 
           self.readToken() # ':-'
           subj = self.term()
           if self.readToken() != ";": 
              raise self.syntaxError("Expecting ';'", self.pos-1)
        if self.tokens[self.pos] != ']': 
            popair = self.popair(subj)
            while self.tokens[self.pos] == ';': 
                self.readToken()
                popair = self.popair(subj)
        if self.readToken() != "]": 
            raise self.syntaxError("Expecting ']'", self.pos-1)
        return subj

    def popair(self, subj): 
       """pred objects"""
       pred = self.pred()
       object = self.objects(subj, pred)

    def pred(self): 
        """(expr) | ('is' expr 'of') | ('has' expr)"""
        t = self.tokens[self.pos]
        if t not in ('is', 'has'): 
            return (1, self.expr())
        elif t == 'is': 
            self.readToken() # 'is'
            expr = self.expr()
            self.readToken() # 'of'
            return (-1, expr)
        elif t == 'has': 
            self.readToken() # 'has'
            return (1, self.expr())

    def objects(self, subj, pred): 
        """term ( ',' term )*"""
        self.gotStatement(subj, pred, self.term())
        while self.tokens[self.pos] == ',': 
            self.readToken()
            self.gotStatement(subj, pred, self.term())
        return None

    def term(self): 
        """expr | name"""
        t = self.tokens[self.pos]
        if ((t not in ('this', '{')) and (not LITERALBmatch(t)) and 
            (not LITERALAmatch(t))): return self.expr()
        else: return self.name()

    def name(self): 
        """'this' | LITERALB | LITERALA | formula"""
        t = self.tokens[self.pos]
        if t == 'this': 
            self.readToken()
            return self.scp
        elif LITERALBmatch(t): 
            return self.strlit(self.readToken())
        elif LITERALAmatch(t): 
            return self.strlit(self.readToken())
        elif t == '{': return self.formula()
        else: raise self.syntaxError("Not a valid literal, '{', or 'this'")

    def expr(self): 
        """URIREF | QNAME | EXIVAR | UNIVAR | PATH | 'a' | '=' 
           | '=>' | list | phrase"""
        t = self.tokens[self.pos]
        if URIREFmatch(t): return self.uriref(self.readToken())
        elif QNAMEmatch(t): return self.qname(self.readToken())
        elif EXIVARmatch(t): return self.bNode(self.readToken())
        elif UNIVARmatch(t): return self.univar(self.readToken())
        elif USE_PATHS: 
            if PATHmatch(t): return self.path(self.readToken())
        elif t == 'a': 
            self.readToken()
            return RDF_type
        elif t == '=':
            self.readToken()
            return DAML_equivalentTo
        elif t == '=>': 
            self.readToken()
            return LOG_implies
        elif t == '(': return self.list()
        elif t == '[': return self.phrase()
        else: raise self.syntaxError("Not a valid URI/QName/bNode/univar/" \
                                     "keyword/start of list or clause")

    def list(self): 
        """'(' term* ')'"""
        self.readToken() # '('
        members = []
        while self.tokens[self.pos] != ')': 
            term = self.term()
            members.append(term)
        self.readToken() # ')'
        return self.makeList(self.something("list", 1), members)

    def formula(self):
        """'{' [ clause ('.' clause)* [ '.' ] ] '}'"""
        # Note that notation3.SinkClass has a formula method, but it 
        # isn't used by any of the SWAP code

        self.readToken() # '{'

        # Make the parent formula the current scope, then generate a 
        # new scope and record the parent of that scope as being the 
        # current parentFormula
        self.parentFormula = self.scp[:]
        self.scp = thisFormula = self.newFormula()
        self._parent[self.scp] = self.parentFormula

        if self.tokens[self.pos] == '}': # allow empty formulae
            self.readToken()
        else: 
            clause = self.clause()
            while self.tokens[self.pos] == '.': 
                if self.readToken() != '.': raise "What?" # it will be a '.'
                if self.tokens[self.pos] == '}': break
                clause = self.clause()
            if self.readToken() != '}': 
                raise self.syntaxError("Expected }", self.pos-1)

        # Revert back to the parent of this formula
        self.scp = self._parent[self.scp]
        return thisFormula

    def qname(self, qname): 
        prefix, name = qname.split(':')
        try: ns = self._bindings[prefix]
        except KeyError: 
            raise self.syntaxError("prefix %s not bound" % prefix)
        else: return (URI, ns + name)

    def bNode(self, s): 
        label = s[2:]
        if label in self._bNodes.keys(): 
            return self._bNodes[label]
        else: return self.something(label)

    def univar(self, s): 
        label = s[1:]
        if label in self._univars.keys(): 
            return self._univars[label]
        else: return self.something(label, quant='forAll')

    def uriref(self, s): 
        return (URI, uripath.join(self._baseURI, s[1:-1]))

    def path(self, s): 
        if USE_PATH: 
            pt = r'(%s|%s|%s|%s|%s|a|\.|\^)' % (LITERALA, URIREF, QNAME, \
                      EXIVAR, UNIVAR)
            terms = re.compile(pt).findall(s)
            obj, terms = self.pathTerm(terms[0]), terms[1:]
            for i in range(len(terms)): 
                # "(i % 2) != 0" is courtesy of the "stupid hacks" department...
                if (i % 2) != 0: 
                    subj = obj # new subject is old object
                    pred = self.pathTerm(terms[i])
                    obj = self.something("path", 1)
                    if terms[i-1] == '.': dr = 0
                    elif terms[i-1] == '^': dr = -1
                    self.gotStatement(subj, (dr, pred), obj)
            return obj

    def pathTerm(self, t): 
        if USE_PATH: 
            if URIREFmatch(t): return self.uriref(t)
            elif LITERALAmatch(t): return self.strlit(t)
            elif QNAMEmatch(t): return self.qname(t)
            elif EXIVARmatch(t): return self.bNode(t)
            elif UNIVARmatch(t): return self.univar(t)
            elif t == 'a': return RDF_type

    def strlit(self, s): 
        """Unescape a Notation3 Unicode string."""
        # @@ use the N-Triples stuff?
        unescapes = {'\\a': '\a', 
                     '\\b': '\b', 
                     '\\f': '\f', 
                     '\\n': '\n', 
                     '\\r': '\r', 
                     '\\t': '\t', 
                     '\\v': '\v'}

        # Consider it to be UTF-8 encoded Unicode
        s = unicode(s, 'utf-8')

        if s.startswith('"""'): s = s[3:-3]
        else: s = s[1:-1]

        s = re.sub(ur'\\u(....)', lambda m: unichr(int(m.group(1), 16)), s)
        for k in unescapes.keys(): s = s.replace(k, unescapes[k])
        s = s.replace('\\\\', '\\')
        return (LIT, s)

    def newFormula(self): 
        self._serial += 1
        label = "formula%s" % self._serial
        baseURI = re.sub(r'\s', '', base64.encodestring(self._baseURI))
        uri = 'formulae:%s#%s' % (baseURI, label)
        return (FORMULA, uri)

    def makeList(self, list, members): 
        if len(members) == 0: return DAML_nil
        elif len(members) == 1: 
           self.gotStatement(list, (0, DAML_first), members[0])
           self.gotStatement(list, (0, DAML_rest), DAML_nil)
           return list
        else: 
           first = list[:]
           for i in range(len(members)): 
              self.gotStatement(first, (0, DAML_first), members[i])
              if i != (len(members)-1): 
                 rest = self.something("rest", 1)
                 self.gotStatement(first, (0, DAML_rest), rest)
                 first = rest[:]
           self.gotStatement(rest, (0, DAML_rest), DAML_nil)
           return list

    def gotStatement(self, subj, pred, obj): 
        dir, pred = pred
        if dir < 0: subj, obj = obj, subj

        if len(self._vars) > 0: 
           try: subj = self._vars[subj]
           except KeyError: pass
           try: pred = self._vars[pred]
           except KeyError: pass
           try: obj = self._vars[obj]
           except KeyError: pass

        if pred not in (LOG_forAll, LOG_forSome): 
            self._sink.makeStatement((subj, pred, obj, self.scp))
        else: 
            if pred == LOG_forAll: q = 'forAll'
            elif pred == LOG_forSome: q = 'forSome'
            self.refQuant(subj, obj, q)

    def refQuant(self, scope, term, q): 
        if '#' in term[1]: label = term[1].split('#')[-1]
        elif q == 'forAll': label = 'uni'
        elif q == 'forSome': label = 'exi'
        var = self.something(label, 1, q)
        self._sink.quant(scope, var)
        self._vars[term] = var

    def something(self, hint, serial=None, quant='forSome'): 
        """Produce, register, and return a quantified variable.

           The serial flag means that we want to produce a variable
           with the same label as the hint if possible, but if not 
           we have to map from the old label to the generated one."""
        if quant == 'forSome': 
            if not serial: it = (EXI, hint)
            else: 
               self._serial += 1
               it = (EXI, '%s%s' % (hint, self._serial))
            while it[1] in self._bNodesList: 
                self._serial += 1
                it = (EXI, '%s%s' % (hint, self._serial))
            if not serial: self._bNodes[hint] = it
            self._bNodesList.append(it[1])
            self._sink.quant(self.scp, it)
        elif quant == 'forAll': # try for the name first
            if not serial: it = (UNI, hint)
            else: 
               self._serial += 1
               it = (UNI, '%s%s' % (hint, self._serial))
            while it[1] in self._univarsList: 
                self._serial += 1
                it = (UNI, '%s%s' % (hint, self._serial))
            if not serial: self._univars[hint] = it
            self._univarsList.append(it[1])
            self._sink.quant(self.parentFormula, it)
        return it

class N3Sink(object): 
    """An outline of the methods called by the N3Parser class."""
    def startDoc(self, formula, baseURI): pass
    def endDoc(self): pass
    def bind(self, pfx, val): pass
    def makeComment(self, s): pass
    def quant(self, formula, var): pass
    def makeStatement(self, (subj, pred, obj, scp)): pass

class __TestN3Sink(N3Sink): 
    """An outline of the methods called by the N3Parser class."""
    def setDefaultNamespace(self, nsPair): pass
    def makeStatement(self, (subj, pred, obj, scp)): 
        print subj, pred, obj, scp

class __SWAPSink(N3Sink): 
    """Convert output from an afon sink into a SWAP sink."""

    def __init__(self, swapsink): 
        self._sink = swapsink # some instance of RDFStore.RDFStore
        self._baseURI = ''
        self._rootFormula = (1, 'RootFormula')

    def makeComment(self, s): 
        self._sink.makeComment(s)

    def bind(self, pfx, val): 
        val = self.n(val)
        if pfx == '': self._sink.setDefaultNamespace((val))
        else: self._sink.bind(pfx, val)

    def n(self, n): 
        n = list(n)
        if n[0] == URI: n[0] = 0 # SYMBOL
        elif n[0] == FORMULA: n[0] = 1 # FORMULA
        elif n[0] == LIT: n[0] = 2 # LITERAL
        # elif n[0] == EXI: n[0] = 3 # ANONYMOUS
        # elif n[0] == UNI: n[0] = 4 # VARIABLE
        elif n[0] == EXI: n = [0, self._baseURI+'#_'+n[1]]
        elif n[0] == UNI: n = [0, self._baseURI+'#'+n[1]]
        return tuple(n)

    def startDoc(self, formula, baseURI): 
        # It makes more sense to send the root formula along with 
        # startDoc just in case the sink needs it (which it often does)
        self._rootFormula = formula
        self._baseURI = baseURI
        self._sink.startDoc()

    def endDoc(self): 
        self._sink.endDoc(self.n(self._rootFormula))

    def makeStatement(self, (subj, pred, obj, scp)): 
        subj, pred, obj, scp = self.n(subj), self.n(pred), \
                               self.n(obj), self.n(scp)
        self._sink.makeStatement((scp, pred, subj, obj))

    def quant(self, formula, var): 
        if type(formula) is type(''): formula = (1, formula)
        else: formula = (1, formula[1])
        if var[0] == EXI: 
           pred = LOG_forSome
           scp = formula
        elif var[0] == UNI: 
           pred = LOG_forAll
           scp = self._rootFormula # or parent formula?
        var = self.n(var)
        self.makeStatement((formula, pred, var, scp))

def __oldMain(): 
    import sys
    if len(sys.argv) == 1: print __doc__
    elif sys.argv[1].endswith('-test'): 
        import urllib
        t = '@prefix : <#> .\n:x _:y "blargh phenomic etc.\u203D" .'
        uri = 'data:,'+urllib.quote(t)
        sink = TestN3Sink()
        p = N3Parser(sink, uri)
        p.load(uri)
    else: 
        sink = TestN3Sink()
        p = N3Parser(sink, sys.argv[1])
        p.load(sys.argv[1])

import node, triple

class __TestSink(object): 
   def startDoc(self, *args): 
      formula, uri = args
      print '# Parsed from URI <%s>' % uri
      print '# using root formula <%s>' % formula[1]
      print 

   def endDoc(self): 
      print 
      print '# [EOF]'

   def bind(self, *args): 
      pass # print '# bind', args

   def makeComment(self, *args): 
      print '#', args

   def quant(self, *args): 
      pass # print '# quant', args

   def makeStatement(self, (subj, pred, objt, scp)): 
      isInRootFormula = True
      for term in (subj, pred, objt, scp): 
         if term[0] == FORMULA: 
            if not (term[1].startswith('formulae:') and 
                    term[1].endswith('$@')): 
               isInRootFormula = False

      if isInRootFormula: 
         print self.triple(subj, pred, objt)

   def triple(self, subj, pred, objt): 
      return "%s %s %s ." % tuple(map(self.term, (subj, pred, objt)))

   def term(self, term): 
      if term[0] == URI or term[0] == FORMULA: 
         return node.URI(term[1])
      elif term[0] == EXI: return node.bNode(term[1])
      elif term[0] == LIT: return node.Literal(term[1])
      elif term[0] == UNI: return node.Var(term[1])
      else: raise "Unknown term type"

class Sink(N3Sink): 
   def __init__(self): 
      self.triples = []

   def startDoc(self, *args): pass
   def endDoc(self): pass
   def bind(self, *args): pass
   def makeComment(self, *args): pass
   def quant(self, *args): pass

   def makeStatement(self, (subj, pred, objt, scp)): 
      isInRootFormula = True
      for term in (subj, pred, objt, scp): 
         if term[0] == FORMULA: 
            if not (term[1].startswith('formulae:') and 
                    term[1].endswith('$@')): 
               isInRootFormula = False

      if isInRootFormula: 
         self.triples.append(self.triple(subj, pred, objt))

   def triple(self, subj, pred, objt): 
      return triple.Triple(map(self.term, (subj, pred, objt)))

   def term(self, term): 
      if term[0] == URI or term[0] == FORMULA: 
         return node.URI(term[1])
      elif term[0] == EXI: return node.bNode(term[1])
      elif term[0] == LIT: return node.Literal(term[1])
      elif term[0] == UNI: return node.Var(term[1])
      else: raise "Unknown term type"

DefaultParser = N3Parser
DefaultSink = Sink

def parseN3(s, base=None, sink=None): 
   sink = sink or DefaultSink()
   p = DefaultParser(sink, base or '')
   p.startDoc()
   p.feed(s)
   p.endDoc()
   return sink

def parseURI(uri, sink=None): 
   import urllib
   u = urllib.urlopen(uri)
   s = u.read()
   u.close()
   return parseN3(s, base=uri, sink=sink)

def main(): 
   import sys
   if len(sys.argv) != 2: print __doc__
   else: parseURI(sys.argv[1])

if __name__=="__main__": 
   main()
