#!/usr/local/bin/python
"""
$Id: notation3.py,v 1.200 2007/12/11 21:18:08 syosi Exp $


This module implements a Nptation3 parser, and the final
part of a notation3 serializer.

See also:

Notation 3
http://www.w3.org/DesignIssues/Notation3

Closed World Machine - and RDF Processor
http://www.w3.org/2000/10/swap/cwm

To DO: See also "@@" in comments

- Clean up interfaces
______________________________________________

Module originally by Dan Connolly, includeing notation3
parser and RDF generator. TimBL added RDF stream model
and N3 generation, replaced stream model with use
of common store/formula API.  Yosi Scharf developped
the module, including tests and test harness.

"""


# Python standard libraries
import types, sys
import string
import codecs # python 2-ism; for writing utf-8 in RDF/xml output
import urllib
import re
from warnings import warn

from sax2rdf import XMLtoDOM # Incestuous.. would be nice to separate N3 and XML

# SWAP http://www.w3.org/2000/10/swap
from diag import verbosity, setVerbosity, progress
from uripath import refTo, join
import uripath
import RDFSink
from RDFSink import CONTEXT, PRED, SUBJ, OBJ, PARTS, ALL4
from RDFSink import  LITERAL, XMLLITERAL, LITERAL_DT, LITERAL_LANG, ANONYMOUS, SYMBOL
from RDFSink import Logic_NS
import diag
from xmlC14n import Canonicalize

from why import BecauseOfData, becauseSubexpression

N3_forSome_URI = RDFSink.forSomeSym
N3_forAll_URI = RDFSink.forAllSym

# Magic resources we know about


from RDFSink import RDF_type_URI, RDF_NS_URI, DAML_sameAs_URI, parsesTo_URI
from RDFSink import RDF_spec, List_NS, uniqueURI
from local_decimal import Decimal

ADDED_HASH = "#"  # Stop where we use this in case we want to remove it!
# This is the hash on namespace URIs

RDF_type = ( SYMBOL , RDF_type_URI )
DAML_sameAs = ( SYMBOL, DAML_sameAs_URI )

from RDFSink import N3_first, N3_rest, N3_nil, N3_li, N3_List, N3_Empty

LOG_implies_URI = "http://www.w3.org/2000/10/swap/log#implies"

INTEGER_DATATYPE = "http://www.w3.org/2001/XMLSchema#integer"
FLOAT_DATATYPE = "http://www.w3.org/2001/XMLSchema#double"
DECIMAL_DATATYPE = "http://www.w3.org/2001/XMLSchema#decimal"
BOOLEAN_DATATYPE = "http://www.w3.org/2001/XMLSchema#boolean"

option_noregen = 0   # If set, do not regenerate genids on output

# @@ I18n - the notname chars need extending for well known unicode non-text
# characters. The XML spec switched to assuming unknown things were name
# characaters.
# _namechars = string.lowercase + string.uppercase + string.digits + '_-'
_notQNameChars = "\t\r\n !\"#$%&'()*.,+/;<=>?@[\\]^`{|}~" # else valid qname :-/
_notNameChars = _notQNameChars + ":"  # Assume anything else valid name :-/
_rdfns = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'


N3CommentCharacter = "#"     # For unix script #! compatabilty

########################################## Parse string to sink
#
# Regular expressions:
eol = re.compile(r'[ \t]*(#[^\n]*)?\r?\n')      # end  of line, poss. w/comment
eof = re.compile(r'[ \t]*(#[^\n]*)?$')          # end  of file, poss. w/comment
ws = re.compile(r'[ \t]*')                      # Whitespace not including NL
signed_integer = re.compile(r'[-+]?[0-9]+')     # integer
number_syntax = re.compile(r'(?P<integer>[-+]?[0-9]+)(?P<decimal>\.[0-9]+)?(?P<exponent>e[-+]?[0-9]+)?')
digitstring = re.compile(r'[0-9]+')             # Unsigned integer      
interesting = re.compile(r'[\\\r\n\"]')
langcode = re.compile(r'[a-zA-Z0-9]+(-[a-zA-Z0-9]+)?')
#"



class SinkParser:
    def __init__(self, store, openFormula=None, thisDoc="", baseURI=None,
                 genPrefix = "", metaURI=None, flags="",
                 why=None):
        """ note: namespace names should *not* end in #;
        the # will get added during qname processing """

        self._bindings = {}
        self._flags = flags
        if thisDoc != "":
            assert ':' in thisDoc, "Document URI not absolute: <%s>" % thisDoc
            self._bindings[""] = thisDoc + "#"  # default

        self._store = store
        if genPrefix: store.setGenPrefix(genPrefix) # pass it on
        
        self._thisDoc = thisDoc
        self.lines = 0              # for error handling
        self.startOfLine = 0        # For calculating character number
        self._genPrefix = genPrefix
        self.keywords = ['a', 'this', 'bind', 'has', 'is', 'of', 'true', 'false' ]
        self.keywordsSet = 0    # Then only can others be considerd qnames
        self._anonymousNodes = {} # Dict of anon nodes already declared ln: Term
        self._variables  = {}
        self._parentVariables = {}
        self._reason = why      # Why the parser was asked to parse this

        self._reason2 = None    # Why these triples
        if diag.tracking: self._reason2 = BecauseOfData(
                        store.newSymbol(thisDoc), because=self._reason) 

        if baseURI: self._baseURI = baseURI
        else:
            if thisDoc:
                self._baseURI = thisDoc
            else:
                self._baseURI = None

        assert not self._baseURI or ':' in self._baseURI

        if not self._genPrefix:
            if self._thisDoc: self._genPrefix = self._thisDoc + "#_g"
            else: self._genPrefix = uniqueURI()

        if openFormula ==None:
            if self._thisDoc:
                self._formula = store.newFormula(thisDoc + "#_formula")
            else:
                self._formula = store.newFormula()
        else:
            self._formula = openFormula
        

        self._context = self._formula
        self._parentContext = None
        
        if metaURI:
            self.makeStatement((SYMBOL, metaURI), # relate doc to parse tree
                            (SYMBOL, PARSES_TO_URI ), #pred
                            (SYMBOL, thisDoc),  #subj
                            self._context)                      # obj
            self.makeStatement(((SYMBOL, metaURI), # quantifiers - use inverse?
                            (SYMBOL, N3_forSome_URI), #pred
                            self._context,  #subj
                            subj))                      # obj

    def here(self, i):
        """String generated from position in file
        
        This is for repeatability when refering people to bnodes in a document.
        This has diagnostic uses less formally, as it should point one to which 
        bnode the arbitrary identifier actually is. It gives the
        line and character number of the '[' charcacter or path character
        which introduced the blank node. The first blank node is boringly _L1C1.
        It used to be used only for tracking, but for tests in general
        it makes the canonical ordering of bnodes repeatable."""

        return "%s_L%iC%i" % (self._genPrefix , self.lines,
                                            i - self.startOfLine + 1) 
        
    def formula(self):
        return self._formula
    
    def loadStream(self, stream):
        return self.loadBuf(stream.read())   # Not ideal

    def loadBuf(self, buf):
        """Parses a buffer and returns its top level formula"""
        self.startDoc()
        self.feed(buf)
        return self.endDoc()    # self._formula


    def feed(self, octets):
        """Feed an octet stream tothe parser
        
        if BadSyntax is raised, the string
        passed in the exception object is the
        remainder after any statements have been parsed.
        So if there is more data to feed to the
        parser, it should be straightforward to recover."""
        str = octets.decode('utf-8')
        i = 0
        while i >= 0:
            j = self.skipSpace(str, i)
            if j<0: return

            i = self.directiveOrStatement(str,j)
            if i<0:
                print "# next char: ", `str[j]` 
                raise BadSyntax(self._thisDoc, self.lines, str, j,
                                    "expected directive or statement")

    def directiveOrStatement(self, str,h):
    
            i = self.skipSpace(str, h)
            if i<0: return i   # EOF

            j = self.directive(str, i)
            if j>=0: return  self.checkDot(str,j)
            
            j = self.statement(str, i)
            if j>=0: return self.checkDot(str,j)
            
            return j


    #@@I18N
    global _notNameChars
    #_namechars = string.lowercase + string.uppercase + string.digits + '_-'
        
    def tok(self, tok, str, i):
        """Check for keyword.  Space must have been stripped on entry and
        we must not be at end of file."""
        
        assert tok[0] not in _notNameChars # not for punctuation
        # was: string.whitespace which is '\t\n\x0b\x0c\r \xa0' -- not ascii
        whitespace = '\t\n\x0b\x0c\r ' 
        if str[i:i+1] == "@":
            i = i+1
        else:
            if tok not in self.keywords:
                return -1   # No, this has neither keywords declaration nor "@"

        if (str[i:i+len(tok)] == tok
            and (str[i+len(tok)] in  _notQNameChars )): 
            i = i + len(tok)
            return i
        else:
            return -1

    def directive(self, str, i):
        j = self.skipSpace(str, i)
        if j<0: return j # eof
        res = []
        
        j = self.tok('bind', str, i)        # implied "#". Obsolete.
        if j>0: raise BadSyntax(self._thisDoc, self.lines, str, i,
                                "keyword bind is obsolete: use @prefix")

        j = self.tok('keywords', str, i)
        if j>0:
            i = self.commaSeparatedList(str, j, res, self.bareWord)
            if i < 0:
                raise BadSyntax(self._thisDoc, self.lines, str, i,
                    "'@keywords' needs comma separated list of words")
            self.setKeywords(res[:])
            if diag.chatty_flag > 80: progress("Keywords ", self.keywords)
            return i


        j = self.tok('forAll', str, i)
        if j > 0:
            i = self.commaSeparatedList(str, j, res, self.uri_ref2)
            if i <0: raise BadSyntax(self._thisDoc, self.lines, str, i,
                        "Bad variable list after @forAll")
            for x in res:
                #self._context.declareUniversal(x)
                if x not in self._variables or x in self._parentVariables:
                    self._variables[x] =  self._context.newUniversal(x)
            return i

        j = self.tok('forSome', str, i)
        if j > 0:
            i = self. commaSeparatedList(str, j, res, self.uri_ref2)
            if i <0: raise BadSyntax(self._thisDoc, self.lines, str, i,
                    "Bad variable list after @forSome")
            for x in res:
                self._context.declareExistential(x)
            return i


        j=self.tok('prefix', str, i)   # no implied "#"
        if j>=0:
            t = []
            i = self.qname(str, j, t)
            if i<0: raise BadSyntax(self._thisDoc, self.lines, str, j,
                                "expected qname after @prefix")
            j = self.uri_ref2(str, i, t)
            if j<0: raise BadSyntax(self._thisDoc, self.lines, str, i,
                                "expected <uriref> after @prefix _qname_")
            ns = self.uriOf(t[1])

            if self._baseURI:
                ns = join(self._baseURI, ns)
            elif ":" not in ns:
                 raise BadSyntax(self._thisDoc, self.lines, str, j,
                    "With no base URI, cannot use relative URI in @prefix <"+ns+">")
            assert ':' in ns # must be absolute
            self._bindings[t[0][0]] = ns
            self.bind(t[0][0], hexify(ns))
            return j

        j=self.tok('base', str, i)      # Added 2007/7/7
        if j >= 0:
            t = []
            i = self.uri_ref2(str, j, t)
            if i<0: raise BadSyntax(self._thisDoc, self.lines, str, j,
                                "expected <uri> after @base ")
            ns = self.uriOf(t[0])

            if self._baseURI:
                ns = join(self._baseURI, ns)
            elif ':' not in ns:
                raise BadSyntax(self._thisDoc, self.lines, str, j,
                    "With no previous base URI, cannot use relative URI in @base  <"+ns+">")
            assert ':' in ns # must be absolute
            self._baseURI = ns
            return i

        return -1  # Not a directive, could be something else.

    def bind(self, qn, uri):
        assert isinstance(uri,
                    types.StringType), "Any unicode must be %x-encoded already"
        if qn == "":
            self._store.setDefaultNamespace(uri)
        else:
            self._store.bind(qn, uri)

    def setKeywords(self, k):
        "Takes a list of strings"
        if k == None:
            self.keywordsSet = 0
        else:
            self.keywords = k
            self.keywordsSet = 1


    def startDoc(self):
        self._store.startDoc()

    def endDoc(self):
        """Signal end of document and stop parsing. returns formula"""
        self._store.endDoc(self._formula)  # don't canonicalize yet
        return self._formula

    def makeStatement(self, quadruple):
        #$$$$$$$$$$$$$$$$$$$$$
#        print "# Parser output: ", `quadruple`
        self._store.makeStatement(quadruple, why=self._reason2)



    def statement(self, str, i):
        r = []

        i = self.object(str, i, r)  #  Allow literal for subject - extends RDF 
        if i<0: return i

        j = self.property_list(str, i, r[0])

        if j<0: raise BadSyntax(self._thisDoc, self.lines,
                                    str, i, "expected propertylist")
        return j

    def subject(self, str, i, res):
        return self.item(str, i, res)

    def verb(self, str, i, res):
        """ has _prop_
        is _prop_ of
        a
        =
        _prop_
        >- prop ->
        <- prop -<
        _operator_"""

        j = self.skipSpace(str, i)
        if j<0:return j # eof
        
        r = []

        j = self.tok('has', str, i)
        if j>=0:
            i = self.prop(str, j, r)
            if i < 0: raise BadSyntax(self._thisDoc, self.lines,
                                str, j, "expected property after 'has'")
            res.append(('->', r[0]))
            return i

        j = self.tok('is', str, i)
        if j>=0:
            i = self.prop(str, j, r)
            if i < 0: raise BadSyntax(self._thisDoc, self.lines, str, j,
                                "expected <property> after 'is'")
            j = self.skipSpace(str, i)
            if j<0:
                raise BadSyntax(self._thisDoc, self.lines, str, i,
                            "End of file found, expected property after 'is'")
                return j # eof
            i=j
            j = self.tok('of', str, i)
            if j<0: raise BadSyntax(self._thisDoc, self.lines, str, i,
                                "expected 'of' after 'is' <prop>")
            res.append(('<-', r[0]))
            return j

        j = self.tok('a', str, i)
        if j>=0:
            res.append(('->', RDF_type))
            return j

            
        if str[i:i+2] == "<=":
            res.append(('<-', self._store.newSymbol(Logic_NS+"implies")))
            return i+2

        if str[i:i+1] == "=":
            if str[i+1:i+2] == ">":
                res.append(('->', self._store.newSymbol(Logic_NS+"implies")))
                return i+2
            res.append(('->', DAML_sameAs))
            return i+1

        if str[i:i+2] == ":=":
            # patch file relates two formulae, uses this    @@ really?
            res.append(('->', Logic_NS+"becomes")) 
            return i+2

        j = self.prop(str, i, r)
        if j >= 0:
            res.append(('->', r[0]))
            return j

        if str[i:i+2] == ">-" or str[i:i+2] == "<-":
            raise BadSyntax(self._thisDoc, self.lines, str, j,
                                        ">- ... -> syntax is obsolete.")

        return -1

    def prop(self, str, i, res):
        return self.item(str, i, res)

    def item(self, str, i, res):
        return self.path(str, i, res)

    def blankNode(self, uri=None):
        if "B" not in self._flags:
            return self._context.newBlankNode(uri, why=self._reason2)
        x = self._context.newSymbol(uri)
        self._context.declareExistential(x)
        return x
        
    def path(self, str, i, res):
        """Parse the path production.
        """
        j = self.nodeOrLiteral(str, i, res)
        if j<0: return j  # nope

        while str[j:j+1] in "!^.":  # no spaces, must follow exactly (?)
            ch = str[j:j+1]     # @@ Allow "." followed IMMEDIATELY by a node.
            if ch == ".":
                ahead = str[j+1:j+2]
                if not ahead or (ahead in _notNameChars
                            and ahead not in ":?<[{("): break
            subj = res.pop()
            obj = self.blankNode(uri=self.here(j))
            j = self.node(str, j+1, res)
            if j<0: raise BadSyntax(self._thisDoc, self.lines, str, j,
                            "EOF found in middle of path syntax")
            pred = res.pop()
            if ch == "^": # Reverse traverse
                self.makeStatement((self._context, pred, obj, subj)) 
            else:
                self.makeStatement((self._context, pred, subj, obj)) 
            res.append(obj)
        return j

    def anonymousNode(self, ln):
        """Remember or generate a term for one of these _: anonymous nodes"""
        term = self._anonymousNodes.get(ln, None)
        if term != None: return term
        term = self._store.newBlankNode(self._context, why=self._reason2)
        self._anonymousNodes[ln] = term
        return term

    def node(self, str, i, res, subjectAlready=None):
        """Parse the <node> production.
        Space is now skipped once at the beginning
        instead of in multipe calls to self.skipSpace().
        """
        subj = subjectAlready

        j = self.skipSpace(str,i)
        if j<0: return j #eof
        i=j
        ch = str[i:i+1]  # Quick 1-character checks first:

        if ch == "[":
            bnodeID = self.here(i)
            j=self.skipSpace(str,i+1)
            if j<0: raise BadSyntax(self._thisDoc,
                                    self.lines, str, i, "EOF after '['")
            if str[j:j+1] == "=":     # Hack for "is"  binding name to anon node
                i = j+1
                objs = []
                j = self.objectList(str, i, objs);
                if j>=0:
                    subj = objs[0]
                    if len(objs)>1:
                        for obj in objs:
                            self.makeStatement((self._context,
                                                DAML_sameAs, subj, obj))
                    j = self.skipSpace(str, j)
                    if j<0: raise BadSyntax(self._thisDoc, self.lines, str, i,
                        "EOF when objectList expected after [ = ")
                    if str[j:j+1] == ";":
                        j=j+1
                else:
                    raise BadSyntax(self._thisDoc, self.lines, str, i,
                                        "objectList expected after [= ")

            if subj is None:
                subj=self.blankNode(uri= bnodeID)

            i = self.property_list(str, j, subj)
            if i<0: raise BadSyntax(self._thisDoc, self.lines, str, j,
                                "property_list expected")

            j = self.skipSpace(str, i)
            if j<0: raise BadSyntax(self._thisDoc, self.lines, str, i,
                "EOF when ']' expected after [ <propertyList>")
            if str[j:j+1] != "]":
                raise BadSyntax(self._thisDoc,
                                    self.lines, str, j, "']' expected")
            res.append(subj)
            return j+1

        if ch == "{":
            ch2 = str[i+1:i+2]
            if ch2 == '$':
                i += 1
                j = i + 1
                List = []
                first_run = True
                while 1:
                    i = self.skipSpace(str, j)
                    if i<0: raise BadSyntax(self._thisDoc, self.lines, str, i,
                                                    "needed '$}', found end.")                    
                    if str[i:i+2] == '$}':
                        j = i+2
                        break

                    if not first_run:
                        if str[i:i+1] == ',':
                            i+=1
                        else:
                            raise BadSyntax(self._thisDoc, self.lines,
                                                str, i, "expected: ','")
                    else: first_run = False
                    
                    item = []
                    j = self.item(str,i, item) #@@@@@ should be path, was object
                    if j<0: raise BadSyntax(self._thisDoc, self.lines, str, i,
                                            "expected item in set or '$}'")
                    List.append(self._store.intern(item[0]))
                res.append(self._store.newSet(List, self._context))
                return j
            else:
                j=i+1
                oldParentContext = self._parentContext
                self._parentContext = self._context
                parentAnonymousNodes = self._anonymousNodes
                grandParentVariables = self._parentVariables
                self._parentVariables = self._variables
                self._anonymousNodes = {}
                self._variables = self._variables.copy()
                reason2 = self._reason2
                self._reason2 = becauseSubexpression
                if subj is None: subj = self._store.newFormula()
                self._context = subj
                
                while 1:
                    i = self.skipSpace(str, j)
                    if i<0: raise BadSyntax(self._thisDoc, self.lines,
                                    str, i, "needed '}', found end.")
                    
                    if str[i:i+1] == "}":
                        j = i+1
                        break
                    
                    j = self.directiveOrStatement(str,i)
                    if j<0: raise BadSyntax(self._thisDoc, self.lines,
                                    str, i, "expected statement or '}'")

                self._anonymousNodes = parentAnonymousNodes
                self._variables = self._parentVariables
                self._parentVariables = grandParentVariables
                self._context = self._parentContext
                self._reason2 = reason2
                self._parentContext = oldParentContext
                res.append(subj.close())   #  No use until closed
                return j

        if ch == "(":
            thing_type = self._store.newList
            ch2 = str[i+1:i+2]
            if ch2 == '$':
                thing_type = self._store.newSet
                i += 1
            j=i+1

            List = []
            while 1:
                i = self.skipSpace(str, j)
                if i<0: raise BadSyntax(self._thisDoc, self.lines,
                                    str, i, "needed ')', found end.")                    
                if str[i:i+1] == ')':
                    j = i+1
                    break

                item = []
                j = self.item(str,i, item) #@@@@@ should be path, was object
                if j<0: raise BadSyntax(self._thisDoc, self.lines, str, i,
                                        "expected item in list or ')'")
                List.append(self._store.intern(item[0]))
            res.append(thing_type(List, self._context))
            return j

        j = self.tok('this', str, i)   # This context
        if j>=0:
            warn(''.__class__(BadSyntax(self._thisDoc, self.lines, str, i,
                "Keyword 'this' was ancient N3. Now use @forSome and @forAll keywords.")))
            res.append(self._context)
            return j

        #booleans
        j = self.tok('true', str, i)
        if j>=0:
            res.append(True)
            return j
        j = self.tok('false', str, i)
        if j>=0:
            res.append(False)
            return j

        if subj is None:   # If this can be a named node, then check for a name.
            j = self.uri_ref2(str, i, res)
            if j >= 0:
                return j

        return -1
        
    def property_list(self, str, i, subj):
        """Parse property list
        Leaves the terminating punctuation in the buffer
        """
        while 1:
            j = self.skipSpace(str, i)
            if j<0:
                raise BadSyntax(self._thisDoc, self.lines, str, i,
                            "EOF found when expected verb in property list")
                return j #eof

            if str[j:j+2] ==":-":
                i = j + 2
                res = []
                j = self.node(str, i, res, subj)
                if j<0: raise BadSyntax(self._thisDoc, self.lines, str, i,
                                        "bad {} or () or [] node after :- ")
                i=j
                continue
            i=j
            v = []
            j = self.verb(str, i, v)
            if j<=0:
                return i # void but valid

            objs = []
            i = self.objectList(str, j, objs)
            if i<0: raise BadSyntax(self._thisDoc, self.lines, str, j,
                                                        "objectList expected")
            for obj in objs:
                dir, sym = v[0]
                if dir == '->':
                    self.makeStatement((self._context, sym, subj, obj))
                else:
                    self.makeStatement((self._context, sym, obj, subj))

            j = self.skipSpace(str, i)
            if j<0:
                raise BadSyntax(self._thisDoc, self.lines, str, j,
                                                "EOF found in list of objects")
                return j #eof
            if str[i:i+1] != ";":
                return i
            i = i+1 # skip semicolon and continue

    def commaSeparatedList(self, str, j, res, what):
        """return value: -1 bad syntax; >1 new position in str
        res has things found appended
        """
        i = self.skipSpace(str, j)
        if i<0:
            raise BadSyntax(self._thisDoc, self.lines, str, i,
                                    "EOF found expecting comma sep list")
            return i
        if str[i] == ".": return j  # empty list is OK
        i = what(str, i, res)
        if i<0: return -1
        
        while 1:
            j = self.skipSpace(str, i)
            if j<0: return j # eof
            ch = str[j:j+1]  
            if ch != ",":
                if ch != ".":
                    return -1
                return j    # Found  but not swallowed "."
            i = what(str, j+1, res)
            if i<0:
                raise BadSyntax(self._thisDoc, self.lines, str, i,
                                                "bad list content")
                return i

    def objectList(self, str, i, res):
        i = self.object(str, i, res)
        if i<0: return -1
        while 1:
            j = self.skipSpace(str, i)
            if j<0:
                raise BadSyntax(self._thisDoc, self.lines, str, j,
                                    "EOF found after object")
                return j #eof
            if str[j:j+1] != ",":
                return j    # Found something else!
            i = self.object(str, j+1, res)
            if i<0: return i

    def checkDot(self, str, i):
            j = self.skipSpace(str, i)
            if j<0: return j #eof
            if str[j:j+1] == ".":
                return j+1  # skip
            if str[j:j+1] == "}":
                return j     # don't skip it
            if str[j:j+1] == "]":
                return j
            raise BadSyntax(self._thisDoc, self.lines,
                    str, j, "expected '.' or '}' or ']' at end of statement")
            return i


    def uri_ref2(self, str, i, res):
        """Generate uri from n3 representation.

        Note that the RDF convention of directly concatenating
        NS and local name is now used though I prefer inserting a '#'
        to make the namesapces look more like what XML folks expect.
        """
        qn = []
        j = self.qname(str, i, qn)
        if j>=0:
            pfx, ln = qn[0]
            if pfx is None:
                assert 0, "not used?"
                ns = self._baseURI + ADDED_HASH
            else:
                try:
                    ns = self._bindings[pfx]
                except KeyError:
                    if pfx == "_":  # Magic prefix 2001/05/30, can be overridden
                        res.append(self.anonymousNode(ln))
                        return j
                    raise BadSyntax(self._thisDoc, self.lines, str, i,
                                "Prefix \"%s:\" not bound" % (pfx))
            symb = self._store.newSymbol(ns + ln)
            if symb in self._variables:
                res.append(self._variables[symb])
            else:
                res.append(symb) # @@@ "#" CONVENTION
            if not string.find(ns, "#"):progress(
                        "Warning: no # on namespace %s," % ns)
            return j

        
        i = self.skipSpace(str, i)
        if i<0: return -1

        if str[i] == "?":
            v = []
            j = self.variable(str,i,v)
            if j>0:              #Forget varibles as a class, only in context.
                res.append(v[0])
                return j
            return -1

        elif str[i]=="<":
            i = i + 1
            st = i
            while i < len(str):
                if str[i] == ">":
                    uref = str[st:i] # the join should dealt with "":
                    if self._baseURI:
                        uref = uripath.join(self._baseURI, uref)
                    else:
                        assert ":" in uref, \
                            "With no base URI, cannot deal with relative URIs"
                    if str[i-1:i]=="#" and not uref[-1:]=="#":
                        uref = uref + "#" # She meant it! Weirdness in urlparse?
                    symb = self._store.newSymbol(uref)
                    if symb in self._variables:
                        res.append(self._variables[symb])
                    else:
                        res.append(symb)
                    return i+1
                i = i + 1
            raise BadSyntax(self._thisDoc, self.lines, str, j,
                            "unterminated URI reference")

        elif self.keywordsSet:
            v = []
            j = self.bareWord(str,i,v)
            if j<0: return -1      #Forget varibles as a class, only in context.
            if v[0] in self.keywords:
                raise BadSyntax(self._thisDoc, self.lines, str, i,
                    'Keyword "%s" not allowed here.' % v[0])
            res.append(self._store.newSymbol(self._bindings[""]+v[0]))
            return j
        else:
            return -1

    def skipSpace(self, str, i):
        """Skip white space, newlines and comments.
        return -1 if EOF, else position of first non-ws character"""
        while 1:
            m = eol.match(str, i)
            if m == None: break
            self.lines = self.lines + 1
            i = m.end()   # Point to first character unmatched
            self.startOfLine = i
        m = ws.match(str, i)
        if m != None:
            i = m.end()
        m = eof.match(str, i)
        if m != None: return -1
        return i

    def variable(self, str, i, res):
        """     ?abc -> variable(:abc)
        """

        j = self.skipSpace(str, i)
        if j<0: return -1

        if str[j:j+1] != "?": return -1
        j=j+1
        i = j
        if str[j] in "0123456789-":
            raise BadSyntax(self._thisDoc, self.lines, str, j,
                            "Varible name can't start with '%s'" % str[j])
            return -1
        while i <len(str) and str[i] not in _notNameChars:
            i = i+1
        if self._parentContext == None:
            raise BadSyntax(self._thisDoc, self.lines, str, j,
                "Can't use ?xxx syntax for variable in outermost level: %s"
                % str[j-1:i])
        varURI = self._store.newSymbol(self._baseURI + "#" +str[j:i])
        if varURI not in self._parentVariables:
            self._parentVariables[varURI] = self._parentContext.newUniversal(varURI
                            , why=self._reason2) 
        res.append(self._parentVariables[varURI])
        return i

    def bareWord(self, str, i, res):
        """     abc -> :abc
        """
        j = self.skipSpace(str, i)
        if j<0: return -1

        if str[j] in "0123456789-" or str[j] in _notNameChars: return -1
        i = j
        while i <len(str) and str[i] not in _notNameChars:
            i = i+1
        res.append(str[j:i])
        return i

    def qname(self, str, i, res):
        """
        xyz:def -> ('xyz', 'def')
        If not in keywords and keywordsSet: def -> ('', 'def')
        :def -> ('', 'def')    
        """

        i = self.skipSpace(str, i)
        if i<0: return -1

        c = str[i]
        if c in "0123456789-+": return -1
        if c not in _notNameChars:
            ln = c
            i = i + 1
            while i < len(str):
                c = str[i]
                if c not in _notNameChars:
                    ln = ln + c
                    i = i + 1
                else: break
        else: # First character is non-alpha
            ln = ''   # Was:  None - TBL (why? useful?)

        if i<len(str) and str[i] == ':':
            pfx = ln
            i = i + 1
            ln = ''
            while i < len(str):
                c = str[i]
                if c not in _notNameChars:
                    ln = ln + c
                    i = i + 1
                else: break

            res.append((pfx, ln))
            return i

        else:  # delimiter was not ":"
            if ln and self.keywordsSet and ln not in self.keywords:
                res.append(('', ln))
                return i
            return -1
            
    def object(self, str, i, res):
        j = self.subject(str, i, res)
        if j>= 0:
            return j
        else:
            j = self.skipSpace(str, i)
            if j<0: return -1
            else: i=j

            if str[i]=='"':
                if str[i:i+3] == '"""': delim = '"""'
                else: delim = '"'
                i = i + len(delim)

                j, s = self.strconst(str, i, delim)

                res.append(self._store.newLiteral(s))
                progress("New string const ", s, j)
                return j
            else:
                return -1

    def nodeOrLiteral(self, str, i, res):
        j = self.node(str, i, res)
        if j>= 0:
            return j
        else:
            j = self.skipSpace(str, i)
            if j<0: return -1
            else: i=j

            ch = str[i]
            if ch in "-+0987654321":
                m = number_syntax.match(str, i)
                if m == None:
                    raise BadSyntax(self._thisDoc, self.lines, str, i,
                                "Bad number syntax")
                j = m.end()
                if m.group('exponent') != None: # includes decimal exponent
                    res.append(float(str[i:j]))
#                   res.append(self._store.newLiteral(str[i:j],
#                       self._store.newSymbol(FLOAT_DATATYPE)))
                elif m.group('decimal') != None:
                    res.append(Decimal(str[i:j]))
                else:
                    res.append(long(str[i:j]))
#                   res.append(self._store.newLiteral(str[i:j],
#                       self._store.newSymbol(INTEGER_DATATYPE)))
                return j

            if str[i]=='"':
                if str[i:i+3] == '"""': delim = '"""'
                else: delim = '"'
                i = i + len(delim)

                dt = None
                j, s = self.strconst(str, i, delim)
                lang = None
                if str[j:j+1] == "@":  # Language?
                    m = langcode.match(str, j+1)
                    if m == None:
                        raise BadSyntax(self._thisDoc, startline, str, i,
                        "Bad language code syntax on string literal, after @")
                    i = m.end()
                    lang = str[j+1:i]
                    j = i
                if str[j:j+2] == "^^":
                    res2 = []
                    j = self.uri_ref2(str, j+2, res2) # Read datatype URI
                    dt = res2[0]
                    if dt.uriref() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral":
                        try:
                            dom = XMLtoDOM('<rdf:envelope xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns">'
                                           + s
                                           + '</rdf:envelope>').firstChild
                        except:
                            raise  ValueError('s="%s"' % s)
                        res.append(self._store.newXMLLiteral(dom))
                        return j
                res.append(self._store.newLiteral(s, dt, lang))
                return j
            else:
                return -1
    
    def uriOf(self, sym):
        if isinstance(sym, types.TupleType):
            return sym[1] # old system for --pipe
        return sym.uriref() # cwm api


    def strconst(self, str, i, delim):
        """parse an N3 string constant delimited by delim.
        return index, val
        """


        j = i
        ustr = u""   # Empty unicode string
        startline = self.lines # Remember where for error messages
        while j<len(str):
            i = j + len(delim)
            if str[j:i] == delim: # done.
                return i, ustr

            if str[j] == '"':
                ustr = ustr + '"'
                j = j + 1
                continue
            m = interesting.search(str, j)  # was str[j:].
            # Note for pos param to work, MUST be compiled  ... re bug?
            assert m , "Quote expected in string at ^ in %s^%s" %(
                str[j-20:j], str[j:j+20]) # we at least have to find a quote

            i = m.start()
            try:
                ustr = ustr + str[j:i]
            except UnicodeError:
                err = ""
                for c in str[j:i]:
                    err = err + (" %02x" % ord(c))
                streason = sys.exc_info()[1].__str__()
                raise BadSyntax(self._thisDoc, startline, str, j,
                "Unicode error appending characters %s to string, because\n\t%s"
                                % (err, streason))
                
#           print "@@@ i = ",i, " j=",j, "m.end=", m.end()

            ch = str[i]
            if ch == '"':
                j = i
                continue
            elif ch == "\r":   # Strip carriage returns
                j = i+1
                continue
            elif ch == "\n":
                if delim == '"':
                    raise BadSyntax(self._thisDoc, startline, str, i,
                                    "newline found in string literal")
                self.lines = self.lines + 1
                ustr = ustr + ch
                j = i + 1
                self.startOfLine = j

            elif ch == "\\":
                j = i + 1
                ch = str[j:j+1]  # Will be empty if string ends
                if not ch:
                    raise BadSyntax(self._thisDoc, startline, str, i,
                                    "unterminated string literal (2)")
                k = string.find('abfrtvn\\"', ch)
                if k >= 0:
                    uch = '\a\b\f\r\t\v\n\\"'[k]
                    ustr = ustr + uch
                    j = j + 1
                elif ch == "u":
                    j, ch = self.uEscape(str, j+1, startline)
                    ustr = ustr + ch
                elif ch == "U":
                    j, ch = self.UEscape(str, j+1, startline)
                    ustr = ustr + ch
                else:
                    raise BadSyntax(self._thisDoc, self.lines, str, i,
                                    "bad escape")

        raise BadSyntax(self._thisDoc, self.lines, str, i,
                        "unterminated string literal")


    def uEscape(self, str, i, startline):
        j = i
        count = 0
        value = 0
        while count < 4:  # Get 4 more characters
            ch = str[j:j+1].lower() 
                # sbp http://ilrt.org/discovery/chatlogs/rdfig/2002-07-05
            j = j + 1
            if ch == "":
                raise BadSyntax(self._thisDoc, startline, str, i,
                                "unterminated string literal(3)")
            k = string.find("0123456789abcdef", ch)
            if k < 0:
                raise BadSyntax(self._thisDoc, startline, str, i,
                                "bad string literal hex escape")
            value = value * 16 + k
            count = count + 1
        uch = unichr(value)
        return j, uch

    def UEscape(self, str, i, startline):
        stringType = type('')
        j = i
        count = 0
        value = '\\U'
        while count < 8:  # Get 8 more characters
            ch = str[j:j+1].lower() 
            # sbp http://ilrt.org/discovery/chatlogs/rdfig/2002-07-05
            j = j + 1
            if ch == "":
                raise BadSyntax(self._thisDoc, startline, str, i,
                                "unterminated string literal(3)")
            k = string.find("0123456789abcdef", ch)
            if k < 0:
                raise BadSyntax(self._thisDoc, startline, str, i,
                                "bad string literal hex escape")
            value = value + ch
            count = count + 1
            
        uch = stringType(value).decode('unicode-escape')
        return j, uch

wide_build = True
try:
    unichr(0x10000)
except ValueError:
    wide_build = False

# If we are going to do operators then they should generate
#  [  is  operator:plus  of (  \1  \2 ) ]


class BadSyntax(SyntaxError):
    def __init__(self, uri, lines, str, i, why):
        self._str = str.encode('utf-8') # Better go back to strings for errors
        self._i = i
        self._why = why
        self.lines = lines
        self._uri = uri 

    def __str__(self):
        str = self._str
        i = self._i
        st = 0
        if i>60:
            pre="..."
            st = i - 60
        else: pre=""
        if len(str)-i > 60: post="..."
        else: post=""

        return 'at line %i of <%s>:\nBad syntax (%s) at ^ in:\n"%s%s^%s%s"' \
               % (self.lines +1, self._uri, self._why, pre,
                                    str[st:i], str[i:i+60], post)



def stripCR(str):
    res = ""
    for ch in str:
        if ch != "\r":
            res = res + ch
    return res

def dummyWrite(x):
    pass

################################################################################


def toBool(s):
    if s == 'true' or s == 'True' or s == '1':
        return True
    if s == 'false' or s == 'False' or s == '0':
        return False
    raise ValueError(s)
    
class ToN3(RDFSink.RDFSink):
    """Serializer output sink for N3
    
      keeps track of most recent subject and predicate reuses them.
      Adapted from Dan's ToRDFParser(Parser);
    """

    flagDocumentation = """Flags for N3 output are as follows:-
        
a   Anonymous nodes should be output using the _: convention (p flag or not).
d   Don't use default namespace (empty prefix)
e   escape literals --- use \u notation
i   Use identifiers from store - don't regen on output
l   List syntax suppression. Don't use (..)
n   No numeric syntax - use strings typed with ^^ syntax
p   Prefix suppression - don't use them, always URIs in <> instead of qnames.
q   Quiet - don't output comments about version and base URI used.
r   Relative URI suppression. Always use absolute URIs.
s   Subject must be explicit for every statement. Don't use ";" shorthand.
t   "this" and "()" special syntax should be suppresed.
u   Use \u for unicode escaping in URIs instead of utf-8 %XX
v   Use  "this log:forAll" for @forAll, and "this log:forAll" for "@forSome".
/   If namespace has no # in it, assume it ends at the last slash if outputting.

Flags for N3 input:

B   Turn any blank node into a existentially qualified explicitly named node.
"""
# "


#   A word about regenerated Ids.
#
# Within the program, the URI of a resource is kept the same, and in fact
# tampering with it would leave risk of all kinds of inconsistencies.
# Hwoever, on output, where there are URIs whose values are irrelevant,
# such as variables and generated IDs from anonymous ndoes, it makes the
# document very much more readable to regenerate the IDs.
#  We use here a convention that underscores at the start of fragment IDs
# are reserved for generated Ids. The caller can change that.
#
# Now there is a new way of generating these, with the "_" prefix
# for anonymous nodes.

    def __init__(self, write, base=None, genPrefix = None,
                            noLists=0 , quiet=0, flags=""):
        gp = genPrefix
        if gp == None:
            gp = "#_g"
            if base!=None: 
                try:
                    gp = uripath.join(base, "#_g")
                except ValueError:
                    pass # bogus: base eg
        RDFSink.RDFSink.__init__(self, gp)
        self._write = self.writeEncoded
        self._writeRaw = write
        self._quiet = quiet or "q" in flags
        self._flags = flags
        self._subj = None
        self.prefixes = {}      # Look up prefix conventions
        self.defaultNamespace = None
        self.indent = 1         # Level of nesting of output
        self.base = base
#       self.nextId = 0         # Regenerate Ids on output
        self.regen = {}         # Mapping of regenerated Ids
        self.noLists = noLists  # Suppress generation of lists?
        self._anodeName = {} # For "a" flag
        self._anodeId = {} # For "a" flag - reverse mapping
        self._needNL = 0    # Do we need to have a newline before a new element?

        if "l" in self._flags: self.noLists = 1
        
    def dummyClone(self):
        "retun a version of myself which will only count occurrences"
        return ToN3(write=dummyWrite, base=self.base, genPrefix=self._genPrefix,
                    noLists=self.noLists, quiet=self._quiet, flags=self._flags )
                    
    def writeEncoded(self, str):
        """Write a possibly unicode string out to the output"""
        try:
            return self._writeRaw(str.encode('utf-8'))
        except UnicodeDecodeError:
            raise UnicodeDecodeError(str, str.__class__)

    def setDefaultNamespace(self, uri):
        return self.bind("", uri)
    
    def bind(self, prefixString, uri):
        """ Just accepting a convention here """
        assert ':' in uri # absolute URI references only
        if "p" in self._flags: return  # Ignore the prefix system completely
#        if not prefixString:
#            raise RuntimError("Please use setDefaultNamespace instead")
        
        if (uri == self.defaultNamespace
            and "d" not in self._flags): return # don't duplicate ??
        self._endStatement()
        self.prefixes[uri] = prefixString
        if 'r' in self._flags:
            self._write(u"@prefix %s: <%s> ."%(prefixString, uri))
        else:
            self._write(u"@prefix %s: <%s> ."%(prefixString, refTo(self.base, uri)))
        self._newline()

    def setDefaultNamespace(self, uri):
        if "d" in self._flags or "p" in self._flags:
            return  # Ignore the prefix system completely
        self._endStatement()
        self.defaultNamespace = uri
        if self.base:  # Sometimes there is none, and now refTo is intolerant
            x = refTo(self.base, uri)
        else:
            x = uri
        self._write(u" @prefix : <%s> ." % x )
        self._newline()
       

    def startDoc(self):
 
        if not self._quiet:  # Suppress stuff which will confuse test diffs
            self._write(u"\n#  Notation3 generation by\n")
            idstr = u"$Id: notation3.py,v 1.200 2007/12/11 21:18:08 syosi Exp $"
            # CVS CHANGES THE ABOVE LINE
            self._write(u"#       " + idstr[5:-2] + u"\n\n") 
            # Strip "$" in case the N3 file is checked in to CVS
            if self.base: self._write(u"#   Base was: " + self.base + u"\n")
        self._write(u"    " * self.indent)
        self._subj = None
#        self._nextId = 0

    def endDoc(self, rootFormulaPair=None):
        self._endStatement()
        self._write(u"\n")
        if self.stayOpen: return  #  fo concatenation
        if not self._quiet: self._write(u"#ENDS\n")
        return  # No formula returned - this is not a store

    def makeComment(self, str):
        for line in string.split(str, "\n"):
            self._write(u"#" + line + "\n")  # Newline order??@@
        self._write(u"    " * self.indent + "    ")


    def _newline(self, extra=0):
        self._write(u"\n"+ u"    " * (self.indent+extra))

    def makeStatement(self, triple, why=None, aIsPossible=1):
#        triple = tuple([a.asPair() for a in triple2])
        if ("a" in self._flags and
            triple[PRED] == (SYMBOL, N3_forSome_URI) and
            triple[CONTEXT] == triple[SUBJ]) :
            # and   # We assume the output is flat @@@ true, we should not
            try:
                aIsPossible = aIsPossible()
            except TypeError:
                aIsPossible = 1
            if aIsPossible:
                ty, value = triple[OBJ]
                i = len(value)
                while i > 0 and value[i-1] not in _notNameChars+"_": i = i - 1
                str2 = value[i:]
                if self._anodeName.get(str2, None) != None:
                    j = 1
                    while 1:
                        str3 = str2 + `j`
                        if self._anodeName.get(str3, None) == None: break
                        j = j +1
                    str2 = str3
                if str2[0] in "0123456789": str2 = "a"+str2
                if diag.chatty_flag > 60: progress(
                                        "Anode %s  means %s" % (str2, value)) 
                self._anodeName[str2] = value
                self._anodeId[value] = str2
                return

        self._makeSubjPred(triple[CONTEXT], triple[SUBJ], triple[PRED])        
        self._write(self.representationOf(triple[CONTEXT], triple[OBJ]))
        self._needNL = 1
                
# Below is for writing an anonymous node

# As object, with one incoming arc:
        
    def startAnonymous(self,  triple):
        self._makeSubjPred(triple[CONTEXT], triple[SUBJ], triple[PRED])
        self._write(u" [")
        self.indent = self.indent + 1
        self._pred = None
        self._newline()
        self._subj = triple[OBJ]    # The object is now the current subject

    def endAnonymous(self, subject, verb):    # Remind me where we are
        self._write(u" ]")
        self.indent = self.indent - 1
        self._subj = subject
        self._pred = verb

# As subject:

    def startAnonymousNode(self, subj):
        if self._subj:
            self._write(u" .")
        self._newline()
        self.indent = self.indent + 1
        self._write(u"  [ ")
        self._subj = subj    # The object is not the subject context
        self._pred = None


    def endAnonymousNode(self, subj=None):    # Remove default subject
        self._write(u" ]")
        if not subj: self._write(u".")
        self.indent = self.indent - 1
        self._newline()
        self._subj = subj
        self._pred = None

# Below we print lists. A list expects to have lots of li links sent

# As subject:

    def startListSubject(self, subj):
        if self._subj:
            self._write(u" .")
        self._newline()
        self.indent = self.indent + 1
        self._write(u"  ( ")
        self._needNL = 0
        self._subj = subj    # The object is not the subject context
        self._pred = N3_li  # expect these until list ends


    def endListSubject(self, subj=None):    # Remove default subject
        self._write(u" )")
        if not subj: self._write(u".")
        self.indent = self.indent - 1
        self._newline()
        self._subj = subj
        self._pred = None


# As Object:

    def startListObject(self,  triple):
        self._makeSubjPred(triple[CONTEXT], triple[SUBJ], triple[PRED])
        self._subj = triple[OBJ]
        self._write(u" (")
        self._needNL = 1      # Choice here of compactness
        self.indent = self.indent + 1
        self._pred = N3_li  # expect these until list ends
        self._subj = triple[OBJ]    # The object is now the current subject

    def endListObject(self, subject, verb):    # Remind me where we are
        self._write(u" )")
        self.indent = self.indent - 1
        self._subj = subject
        self._pred = verb



# Below we print a nested formula of statements

    def startFormulaSubject(self, context):
        if self._subj != context:
            self._endStatement()
        self.indent = self.indent + 1
        self._write(u"{")
        self._newline()
        self._subj = None
        self._pred = None

    def endFormulaSubject(self, subj):    # Remove context
        self._endStatement()     # @@@@@@@@ remove in syntax change to implicit
        self._newline()
        self.indent = self.indent - 1
        self._write(u"}")
        self._subj = subj
        self._pred = None
     
    def startFormulaObject(self, triple):
        self._makeSubjPred(triple[CONTEXT], triple[SUBJ], triple[PRED])
        self.indent = self.indent + 1
        self._write(u"{")
        self._subj = None
        self._pred = None

    def endFormulaObject(self, pred, subj):    # Remove context
        self._endStatement() # @@@@@@@@ remove in syntax change to implicit
        self.indent = self.indent - 1
        self._write(u"}")
#        self._newline()
        self._subj = subj
        self._pred = pred
     
    def _makeSubjPred(self, context, subj, pred):

        if pred == N3_li:
            if  self._needNL:
                self._newline()
            return  # If we are in list mode, don't need to.
        
        varDecl = (subj == context and "v" not in self._flags and (
                pred == (SYMBOL, N3_forAll_URI) or
                pred == (SYMBOL, N3_forSome_URI)))
                
        if self._subj != subj or "s" in self._flags:
            self._endStatement()
            if self.indent == 1:  # Top level only - extra newline
                self._newline()
            if "v" in self._flags or subj != context:
                self._write(self.representationOf(context, subj))
            else:  # "this" suppressed
                if (pred != (SYMBOL, N3_forAll_URI) and
                    pred != (SYMBOL, N3_forSome_URI)):
                    raise ValueError(
                     "On N3 output, 'this' used with bad predicate: %s" % (pred, ))
            self._subj = subj
            self._pred = None

        if self._pred != pred:
            if self._pred:
                if "v" not in self._flags and (
                     self._pred== (SYMBOL, N3_forAll_URI) or
                     self._pred == (SYMBOL, N3_forSome_URI)):
                     self._write(u".")
                else:
                    self._write(u";")
                self._newline(1)   # Indent predicate from subject
            elif not varDecl: self._write(u"    ")

            if varDecl:
                    if pred == (SYMBOL, N3_forAll_URI):
                        self._write( u" @forAll ")
                    else:
                        self._write( u" @forSome ")
            elif pred == (SYMBOL, DAML_sameAs_URI) and "t" not in self._flags:
                self._write(u" = ")
            elif pred == (SYMBOL, RDF_type_URI)  and "t" not in self._flags:
                self._write(u" a ")
            else :
                self._write( u" %s " % self.representationOf(context, pred))
                
            self._pred = pred
        else:
            self._write(u",")
            self._newline(3)    # Same subject and pred => object list

    def _endStatement(self):
        if self._subj:
            self._write(u" .")
            self._newline()
            self._subj = None

    def representationOf(self, context, pair):
        """  Representation of a thing in the output stream

        Regenerates genids if required.
        Uses prefix dictionary to use qname syntax if possible.
        """

        if "t" not in self._flags:
            if pair == context:
                return u"this"
            if pair == N3_nil and not self.noLists:
                return u"()"

        ty, value = pair

        singleLine = "n" in self._flags
        if ty == LITERAL:
            return stringToN3(value, singleLine=singleLine, flags = self._flags)

        if ty == XMLLITERAL:
            st = u''.join([Canonicalize(x, None, unsuppressedPrefixes=['foo']) for x in value.childNodes])
            st = stringToN3(st, singleLine=singleLine, flags=self._flags)
            return st + u"^^" + self.representationOf(context, (SYMBOL,
                    u"http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral"))

        if ty == LITERAL_DT:
            s, dt = value
            if "b" not in self._flags:
                if (dt == BOOLEAN_DATATYPE):
                    return toBool(s) and u"true" or u"false"
            if "n" not in self._flags:
                dt_uri = dt
                if (dt_uri == INTEGER_DATATYPE):
                    return unicode(long(s))
                if (dt_uri == FLOAT_DATATYPE):
                    retVal =  unicode(float(s))    # numeric value python-normalized
                    if 'e' not in retVal:
                        retVal += 'e+00'
                    return retVal
                if (dt_uri == DECIMAL_DATATYPE):
                    retVal = unicode(Decimal(s))
                    if '.' not in retVal:
                        retVal += '.0'
                    return retVal
            st = stringToN3(s, singleLine= singleLine, flags=self._flags)
            return st + u"^^" + self.representationOf(context, (SYMBOL, dt))

        if ty == LITERAL_LANG:
            s, lang = value
            return stringToN3(s, singleLine= singleLine,
                                        flags=self._flags)+ u"@" + lang

        aid = self._anodeId.get(pair[1], None)
        if aid != None:  # "a" flag only
            return u"_:" + aid    # Must start with alpha as per NTriples spec.

        if ((ty == ANONYMOUS)
            and not option_noregen and "i" not in self._flags ):
                x = self.regen.get(value, None)
                if x == None:
                    x = self.genId()
                    self.regen[value] = x
                value = x
#                return "<"+x+">"


        j = string.rfind(value, "#")
        if j<0 and "/" in self._flags:
            j=string.rfind(value, "/")   # Allow "/" namespaces as a second best
        
        if (j>=0
            and "p" not in self._flags):   # Suppress use of prefixes?
            for ch in value[j+1:]:  #  Examples: "." ";"  we can't have in qname
                if ch in _notNameChars:
                    if verbosity() > 20:
                        progress("Cannot have character %i in local name for %s"
                                    % (ord(ch), `value`))
                    break
            else:
                namesp = value[:j+1]
                if (self.defaultNamespace
                    and self.defaultNamespace == namesp
                    and "d" not in self._flags):
                    return u":"+value[j+1:]
                self.countNamespace(namesp)
                prefix = self.prefixes.get(namesp, None) # @@ #CONVENTION
                if prefix != None : return prefix + u":" + value[j+1:]
            
                if value[:j] == self.base:   # If local to output stream,
                    return u"<#" + value[j+1:] + u">" # use local frag id
        
        if "r" not in self._flags and self.base != None:
            value = hexify(refTo(self.base, value))
        elif "u" in self._flags: value = backslashUify(value)
        else: value = hexify(value)

        return u"<" + value + u">"    # Everything else

def nothing():
    pass

import triple_maker as tm

LIST = 10000
QUESTION = 10001
class tmToN3(RDFSink.RDFSink):
    """


    """
    def __init__(self, write, base=None, genPrefix = None,
                                    noLists=0 , quiet=0, flags=""):
        gp = genPrefix
        if gp == None:
            gp = "#_g"
            if base!=None: 
                try:
                    gp = uripath.join(base, "#_g")
                except ValueError:
                    pass # bogus: base eg
        RDFSink.RDFSink.__init__(self, gp)
        self._write = self.writeEncoded
        self._writeRaw = write
        self._quiet = quiet or "q" in flags
        self._flags = flags
        self._subj = None
        self.prefixes = {}      # Look up prefix conventions
        self.defaultNamespace = None
        self.indent = 1         # Level of nesting of output
        self.base = base
#       self.nextId = 0         # Regenerate Ids on output
        self.regen = {}         # Mapping of regenerated Ids
#       self.genPrefix = genPrefix  # Prefix for generated URIs on output
        self.noLists = noLists  # Suppress generation of lists?
        self._anodeName = {} # For "a" flag
        self._anodeId = {} # For "a" flag - reverse mapping

        if "l" in self._flags: self.noLists = 1
    
    def writeEncoded(self, str):
        """Write a possibly unicode string out to the output"""
        return self._writeRaw(str.encode('utf-8'))

    def _newline(self, extra=0):
        self._needNL = 0
        self._write("\n"+ "    " * (self.indent+extra))
        
    def bind(self, prefixString, uri):
        """ Just accepting a convention here """
        assert ':' in uri # absolute URI references only
        if "p" in self._flags: return  # Ignore the prefix system completely
        if not prefixString:
            return self.setDefaultNamespace(uri)
        
        if (uri == self.defaultNamespace
            and "d" not in self._flags): return # don't duplicate ??
        self.endStatement()
        self.prefixes[uri] = prefixString
        self._write(" @prefix %s: <%s> ." %
                            (prefixString, refTo(self.base, uri)) )
        self._newline()

    def setDefaultNamespace(self, uri):
        if "d" in self._flags or "p" in self._flags: return  # no prefix system
        self.endStatement()
        self.defaultNamespace = uri
        if self.base:  # Sometimes there is none, and now refTo is intolerant
            x = refTo(self.base, uri)
        else:
            x = uri
        self._write(" @prefix : <%s> ." % x )
        self._newline()
    
    def start(self):
        pass
        self._parts = [0]
        self._types = [None]
        self._nodeEnded = False

    def end(self):
        self._write('\n\n#End')

    def addNode(self, node):
        self._parts[-1] += 1
        if node is not None:
            self._realEnd()
            if self._types == LITERAL:
                lit, dt, lang = node
                singleLine = "n" in self._flags
                if dt != None and "n" not in self._flags:
                    dt_uri = dt          
                    if (dt_uri == INTEGER_DATATYPE):
                        self._write(str(long(lit)))
                        return
                    if (dt_uri == FLOAT_DATATYPE):
                        self._write(str(float(lit))) # numeric python-normalized
                        return
                    if (dt_uri == DECIMAL_DATATYPE):
                        self._write(str(Decimal(lit)))
                        return
                st = stringToN3(lit, singleLine= singleLine, flags=self._flags)
                if lang != None: st = st + "@" + lang
                if dt != None: st = st + "^^" + self.symbolString(dt)
                self._write(st)
            elif self._types == SYMBOL:
                self._write(self.symbolString(node) + ' ')

            elif self._types == QUESTION:
                self._write('?' + node + ' ')

    def _realEnd(self):
        if self._nodeEnded:
            self._nodeEnded = False
            if self._parts[-1] == 1:
                self._write(' . \n')
            elif self._parts[-1] == 2:
                self._write(';\n')
            elif self._parts[-1] == 3:
                self._write(',\n')
            else:
                pass
    
    def symbolString(self, value):
        j = string.rfind(value, "#")
        if j<0 and "/" in self._flags:
            j=string.rfind(value, "/")   # Allow "/" namespaces as a second best
        
        if (j>=0
            and "p" not in self._flags):   # Suppress use of prefixes?
            for ch in value[j+1:]:  #  Examples: "." ";"  we can't have in qname
                if ch in _notNameChars:
                    if verbosity() > 0:
                        progress("Cannot have character %i in local name."
                                            % ord(ch))
                    break
            else:
                namesp = value[:j+1]
                if (self.defaultNamespace
                    and self.defaultNamespace == namesp
                    and "d" not in self._flags):
                    return ":"+value[j+1:]
                self.countNamespace(namesp)
                prefix = self.prefixes.get(namesp, None) # @@ #CONVENTION
                if prefix != None : return prefix + ":" + value[j+1:]
            
                if value[:j] == self.base:   # If local to output stream,
                    return "<#" + value[j+1:] + ">" #   use local frag id
        
        if "r" not in self._flags and self.base != None:
            value = refTo(self.base, value)
        elif "u" in self._flags: value = backslashUify(value)
        else: value = hexify(value)

        return "<" + value + ">"    # Everything else
        
    def IsOf(self):
        self._write('is ')
        self._predIsOfs[-1] = FRESH

    def checkIsOf(self):
        return self._predIsOfs[-1]

    def forewardPath(self):
        self._write('!')
        
    def backwardPath(self):
        self._write('^')
        
    def endStatement(self):
        self._parts[-1] = 0
        self._nodeEnded = True

    def addLiteral(self, lit, dt=None, lang=None):
        self._types = LITERAL
        self.addNode((lit, dt, lang))

    def addSymbol(self, sym):
        self._types = SYMBOL
        self.addNode(sym)
    
    def beginFormula(self):
        self._realEnd()
        self._parts.append(0)
        self._write('{')

    def endFormula(self):
        self._parts.pop()
        self._write('}')
        self._types = None
        self.addNode(None)

    def beginList(self):
        self._realEnd()
        self._parts.append(-1)
        self._write('(')

    def endList(self):
        self._parts.pop()
        self._types = LIST
        self._write(') ')
        self.addNode(None)

    def addAnonymous(self, Id):
        """If an anonymous shows up more than once, this is the
        function to call

        """
        if Id not in bNodes:
            a = self.formulas[-1].newBlankNode()
            bNodes[Id] = a
        else:
            a = bNodes[Id]
        self.addNode(a)
        
    
    def beginAnonymous(self):
        self._realEnd()
        self._parts.append(0)
        self._write('[')
        

    def endAnonymous(self):
        self._parts.pop()
        self._write(']')
        self._types = None
        self.addNode(None)

    def declareExistential(self, sym):
        self._write('@forSome ' + sym + ' . ')

    def declareUniversal(self, sym):
        self._write('@forAll ' + sym + ' . ')

    def addQuestionMarkedSymbol(self, sym):
        self._types = QUESTION
        self.addNode(sym)

        
###################################################
#
#   Utilities
#

Escapes = {'a':  '\a',
           'b':  '\b',
           'f':  '\f',
           'r':  '\r',
           't':  '\t',
           'v':  '\v',
           'n':  '\n',
           '\\': '\\',
           '"':  '"'}

forbidden1 = re.compile(ur'[\\\"\a\b\f\r\v\u0080-\U0000ffff]')
forbidden2 = re.compile(ur'[\\\"\a\b\f\r\v\t\n\u0080-\U0000ffff]')
#"
def stringToN3(str, singleLine=0, flags=""):
    res = u''
    if (len(str) > 20 and
        str[-1] <> u'"' and
        not singleLine and
        (string.find(str, u"\n") >=0 
         or string.find(str, u'"') >=0)):
        delim= u'"""'
        forbidden = forbidden1   # (allow tabs too now)
    else:
        delim = u'"'
        forbidden = forbidden2
        
    i = 0

    while i < len(str):
        m = forbidden.search(str, i)
        if not m:
            break

        j = m.start()
        res = res + str[i:j]
        ch = m.group(0)
        if ch == u'"' and delim == u'"""' and str[j:j+3] != u'"""':  #"
            res = res + ch
        else:
            k = string.find(u'\a\b\f\r\t\v\n\\"', ch)
            if k >= 0: res = res + u"\\" + u'abfrtvn\\"'[k]
            else:
                if 'e' in flags:
#                res = res + ('\\u%04x' % ord(ch))
                    res = res + (u'\\u%04X' % ord(ch)) 
                    # http://www.w3.org/TR/rdf-testcases/#ntriples
                else:
                    res = res + ch
        i = j + 1

    # The following code fixes things for really high range Unicode
    newstr = u""
    for ch in res + str[i:]:
        if ord(ch)>65535:
            newstr = newstr + (u'\\U%08X' % ord(ch)) 
                # http://www.w3.org/TR/rdf-testcases/#ntriples
        else:
            newstr = newstr + ch
    #

    return delim + newstr + delim

def backslashUify(ustr):
    """Use URL encoding to return an ASCII string corresponding
        to the given unicode"""
#    progress("String is "+`ustr`)
#    s1=ustr.encode('utf-8')
    str  = u""
    for ch in ustr:  # .encode('utf-8'):
        if ord(ch) > 65535:
            ch = u"\\U%08X" % ord(ch)       
        elif ord(ch) > 126:
            ch = u"\\u%04X" % ord(ch)
        else:
            ch = u"%c" % ord(ch)
        str = str + ch
    return str

def hexify(ustr):
    """Use URL encoding to return an ASCII string
    corresponding to the given UTF8 string

    >>> hexify("http://example/a b")
    'http://example/a%20b'
    
    """   #"
#    progress("String is "+`ustr`)
#    s1=ustr.encode('utf-8')
    str  = ""
    for ch in ustr:  # .encode('utf-8'):
        if ord(ch) > 126 or ord(ch) < 33 :
            ch = "%%%02X" % ord(ch)
        else:
            ch = "%c" % ord(ch)
        str = str + ch
    return str
    
def dummy():
        res = ""
        if len(str) > 20 and (string.find(str, "\n") >=0 
                                or string.find(str, '"') >=0):
                delim= '"""'
                forbidden = "\\\"\a\b\f\r\v"    # (allow tabs too now)
        else:
                delim = '"'
                forbidden = "\\\"\a\b\f\r\v\t\n"
        for i in range(len(str)):
                ch = str[i]
                j = string.find(forbidden, ch)
                if ch == '"' and delim == '"""' \
                                and i+1 < len(str) and str[i+1] != '"':
                    j=-1   # Single quotes don't need escaping in long format
                if j>=0: ch = "\\" + '\\"abfrvtn'[j]
                elif ch not in "\n\t" and (ch < " " or ch > "}"):
                    ch = "[[" + `ch` + "]]" #[2:-1] # Use python
                res = res + ch
        return delim + res + delim

def _test():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _test()

#ends

