#!/usr/local/bin/python
"""
$Id: toXML.py,v 1.41 2007/10/15 14:55:55 syosi Exp $


This module implements basic sources and sinks for RDF data.
It defines a stream interface for such data.
It has a command line interface, can work as a web query engine,
and has built in test(), all of which demosntrate how it is used.

To make a new RDF processor, subclass RDFSink.

See also:

Notation 3
http://www.w3.org/DesignIssues/Notation3

Closed World Machine - and RDF Processor
http;//www.w3.org/2000/10/swap/cwm

To DO: See also "@@" in comments

Internationlization:
- Decode incoming N3 file as unicode
- Encode outgoing file
- unicode \u  (??) escapes in parse
- unicode \u  (??) escapes in string output

Note currently unicode strings work in this code
but fail when they are output into the python debugger
interactive window.

______________________________________________

Module originally by Dan Connolly.
TimBL added RDF stream model.


"""



import string
import codecs # python 2-ism; for writing utf-8 in RDF/xml output
import urlparse
import urllib
import re
import sys
#import thing
from uripath import refTo
from diag import progress

from random import choice, seed
seed(23)

from xmlC14n import Canonicalize

import RDFSink
from set_importer import Set

from RDFSink import CONTEXT, PRED, SUBJ, OBJ, PARTS, ALL4
from RDFSink import FORMULA, LITERAL, XMLLITERAL, LITERAL_DT, LITERAL_LANG, ANONYMOUS, SYMBOL
from RDFSink import Logic_NS, NODE_MERGE_URI

from isXML import isXMLChar, NCNameChar, NCNameStartChar, setXMLVersion, getXMLVersion

N3_forSome_URI = RDFSink.forSomeSym
N3_forAll_URI = RDFSink.forAllSym

# Magic resources we know about

RDF_type_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDF_NS_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DAML_NS=DPO_NS = "http://www.daml.org/2001/03/daml+oil#"  # DAML plus oil
DAML_sameAs_URI = DPO_NS+"sameAs"
parsesTo_URI = Logic_NS + "parsesTo"
RDF_spec = "http://www.w3.org/TR/REC-rdf-syntax/"

ADDED_HASH = "#"  # Stop where we use this in case we want to remove it!
# This is the hash on namespace URIs

# Should the internal representation of lists be with DAML:first and :rest?
DAML_LISTS = 1    # Else don't do these - do the funny compact ones- not a good idea after all

RDF_type = ( SYMBOL , RDF_type_URI )
DAML_sameAs = ( SYMBOL, DAML_sameAs_URI )

List_NS = RDF_NS_URI     # We have to pick just one all the time

# For lists:
N3_first = (SYMBOL, List_NS + "first")
N3_rest = (SYMBOL, List_NS + "rest")
RDF_li = (SYMBOL, List_NS + "li")
# N3_only = (SYMBOL, List_NS + "only")
N3_nil = (SYMBOL, List_NS + "nil")
N3_List = (SYMBOL, List_NS + "List")
N3_Empty = (SYMBOL, List_NS + "Empty")

XML_NS_URI = "http://www.w3.org/XML/1998/namespace"



option_noregen = 0   # If set, do not regenerate genids on output


########################## RDF 1.0 Syntax generator

global _namechars       
_namechars = string.lowercase + string.uppercase + string.digits + '_-'
            
def dummyWrite(x):
    pass


class ToRDF(RDFSink.RDFStructuredOutput):
    """keeps track of most recent subject, reuses it"""

    _valChars = string.lowercase + string.uppercase + string.digits + "_ !#$%&().,+*/"
    #@ Not actually complete, and can encode anyway
    def __init__(self, outFp, thisURI=None, base=None, flags=""):
        RDFSink.RDFSink.__init__(self)
        if outFp == None:
            self._xwr = XMLWriter(dummyWrite, self)
        else:
            dummyEnc, dummyDec, dummyReader, encWriter = codecs.lookup('utf-8')
            z = encWriter(outFp)
            zw = z.write
            self._xwr = XMLWriter(zw, self)
        self._subj = None
        self._base = base
        self._formula = None   # Where do we get this from? The outermost formula
        if base == None: self._base = thisURI
        self._thisDoc = thisURI
        self._flags = flags
        self._nodeID = {}
        self._nextnodeID = 0
        self._docOpen = 0  # Delay doc open <rdf:RDF .. till after binds
        def doNothing():
            pass
        self._toDo = doNothing
        self.namespace_redirections = {}
        self.illegals = Set()
        self.stack = [0]

    #@@I18N
    _rdfns = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'

    def dummyClone(self):
        "retun a version of myself which will only count occurrences"
        return ToRDF(None, self._thisDoc, base=self._base, flags=self._flags )

    def bind(self, prefix, namespace):
        if prefix in self.namespace_redirections:
            prefix = self.namespace_redirections[prefix]
        else:
            realPrefix = prefix
            while prefix in self.illegals or prefix[:3] == 'xml':
                prefix = choice(string.ascii_letters) + prefix
            if realPrefix is not prefix:
                self.illegals.add(prefix)
                self.namespace_redirections[realPrefix] = prefix
        return RDFSink.RDFStructuredOutput.bind(self, prefix, namespace)

    def startDoc(self):
        pass

    flagDocumentation = """
Flags to control RDF/XML output (after --rdf=) areas follows:
        
b  - Don't use nodeIDs for Bnodes
c  - Don't use elements as class names
d  - Default namespace supressed.
l  - Don't use RDF collection syntax for lists
r  - Relative URI suppression. Always use absolute URIs.
z  - Allow relative URIs for namespaces

"""

    def endDoc(self, rootFormulaPair=None):
        if self.stayOpen: return # Ignore close if stayOpen is set, (for concatenation)
        self.flushStart()  # Note: can't just leave empty doc if not started: bad XML
        if self._subj:
            self._xwr.endElement()  # </rdf:Description>
        self._subj = None
        self._xwr.endElement()  # </rdf:RDF>
        self._xwr.endDocument()

    def makeComment(self, str):
        if self._docOpen:
            self._xwr.makeComment(str)
            return
        toDo = self._toDo
        def toDoMore():
            toDo()
            self._xwr.makeComment(str)
        self._toDo = toDoMore

    def referenceTo(self, uri):
        "Conditional relative URI"
        if "r" in self._flags or self._base == None:
            return uri
        return refTo(self._base, uri)

    def flushStart(self):
        if not self._docOpen:
            if getXMLVersion() != '1.0':
                self._xwr.makePI('xml version="%s"' % getXMLVersion())
            self._toDo()
            if self.prefixes.get(RDF_NS_URI, ":::") == ":::":
                if self.namespaces.get("rdf", ":::") ==":::":
                    self.bind("rdf", RDF_NS_URI)
##                else:
##                    ns = findLegal(self.prefixes, RDF_NS_URI)
##                    self.bind(ns, RDF_NS_URI)
#            if self.prefixes.get(Logic_NS, ":::") == ":::":
#                if self.namespaces.get("log", ":::") ==":::":
#                    self.bind("log", Logic_NS)
            ats = []
            ps = self.prefixes.values()
            ps.sort()    # Cannonicalize output somewhat
            if self.defaultNamespace and "d" not in self._flags:
                if "z" in self._flags:
                    ats.append(('xmlns',
                        self.referenceTo(self.defaultNamespace)))
                else:
                    ats.append(('xmlns',self.defaultNamespace))
            for pfx in ps:
                nsvalue = self.namespaces[pfx]
                if "z" in self._flags:
                    nsvalue = self.referenceTo( nsvalue)
                ats.append(('xmlns:'+pfx, nsvalue))

            self._xwr.startElement(RDF_NS_URI+'RDF', ats, self.prefixes)
            self._subj = None
            self._nextId = 0
            self._docOpen = 1

    def makeStatement(self,  tuple, why=None, aIsPossible=0):
        context, pred, subj, obj = tuple # Context is ignored


        if self.stack[-1]:
            if pred == RDF_li:
                pass
            elif pred == N3_first:
                pred = RDF_li
            elif pred == RDF_type and obj == N3_List:
                return  # We knew
            elif pred == RDF_type and obj == N3_Empty:
                return  # not how we would have put it but never mind
            elif pred == N3_rest:
                return # Ignore rest
            else: raise RuntimeError ("Should only see %s and %s in list mode, for tuple %s and stack %s" 
                                    %(N3_first, N3_rest, `tuple`, `self.stack`))

        if subj == context: # and context == self._formula:
            if pred == (SYMBOL, N3_forAll_URI):
                progress("Ignoring universal quantification of ", obj)
                return
            elif pred == (SYMBOL, N3_forSome_URI):
                nid = self._nodeID.get(obj, None)
                if nid == None and not("b" in self._flags):
                    self._nextnodeID += 1
                    nid = 'b'+`self._nextnodeID`
                    self._nodeID[obj] = nid
                    #progress("object is now", obj, nid)
                return
            
        if subj[0] not in (SYMBOL, ANONYMOUS, LITERAL, LITERAL_DT, LITERAL_LANG):
            progress("Warning:  because subject is not symbol, bnode or literal, Ignoring ", tuple)
            return
#        progress("@@@ subject node is ", subj)

        self.flushStart()
        if self._formula == None:
            self._formula = context   # Asssume first statement is in outermost context @@
        predn = self.referenceTo( pred[1])
        subjn = self.referenceTo( subj[1])
        
        if pred == (SYMBOL, RDF_NS_URI+'Description'):
            raise ValueError('rdf:Description is not a valid predicate')

        if self._subj != subj:
            if self._subj:
                self._xwr.endElement()
            self._subj = subj
            if (pred == (SYMBOL, RDF_type_URI)# Special case starting with rdf:type as element name
                and obj[0] != LITERAL
                and "c" not in self._flags): # "c" flag suppresses class element syntax on RDF output
#                 self._xwr.startElement(obj[1], [(RDF_NS_URI+" about", subjn),], self.prefixes)
#                 return
                 start_tag = obj[1]
            else:
                 start_tag = RDF_NS_URI+'Description'
#            progress("@@ Start tag: %s" % start_tag)    
            if subj[0] == SYMBOL or subj[0] == ANONYMOUS:
                nid = self._nodeID.get(subj, None)
                if nid == None:
                    if subj[0] == ANONYMOUS:
                        self._xwr.startElement(start_tag,
                                [], self.prefixes)
                    else:
                        self._xwr.startElement(start_tag,
                                [(RDF_NS_URI+" about", subjn),], self.prefixes)
                else:
                    self._xwr.startElement(start_tag,
                                        [(RDF_NS_URI+" nodeID", nid),], self.prefixes)
                if start_tag != RDF_NS_URI+'Description':
                    return
                
            elif subj[0] == LITERAL:
                raise ValueError(
                """Bad subject of statement: %s.
                RDF/XML cannot serialize a graph in which a subject is a literal.""" % subj[1])
                # See http://lists.w3.org/Archives/Public/public-cwm-bugs/2004Aug/0014.html
                v = subj[1]
                attrs = []  # Literal
                if type(v) is type((1,1)):
                    v, dt, lang = v
                    if dt != None: attrs.append((RDF_NS_URI+'datatype', dt))
                    if lang != None: attrs.append((XML_NS_URI+' lang', lang))
                self._xwr.startElement(RDF_NS_URI+'Description',
                                    [], self.prefixes)
                self._xwr.startElement(RDF_NS_URI+"is", attrs, self.prefixes)
                self._xwr.data(v)
                self._xwr.endElement()
            else:
                raise RuntimeError("Unexpected subject", `subj`)

        if obj[0] not in (XMLLITERAL, LITERAL, LITERAL_DT, LITERAL_LANG):
            nid = self._nodeID.get(obj, None)
            if nid == None:
                objn = self.referenceTo( obj[1])
                # progress("@@@ objn=%s, obj[1]=%s" %(objn, obj[1]))
                nid2 = self._nodeID.get(pred, None)
                if nid2 is None:
                    self._xwr.emptyElement(pred[1], [(RDF_NS_URI+' resource', objn)], self.prefixes)
                else:
                    bNodePredicate()
            else:
                self._xwr.emptyElement(pred[1], [(RDF_NS_URI+' nodeID', nid)], self.prefixes)           
            return

        attrs = []  # Literal
        v = obj[1]
        if obj[0] == XMLLITERAL:
            attrs.append((RDF_NS_URI+' parseType', 'Literal'))
        elif obj[0] == LITERAL_DT:
            v, dt = v
            if dt != None: attrs.append((RDF_NS_URI+' datatype', dt))
        elif obj[0] == LITERAL_LANG:
            v, lang = v
            if lang != None: attrs.append((XML_NS_URI+' lang', lang))
        nid = self._nodeID.get(pred, None)
        if nid is None:
            self._xwr.startElement(pred[1], attrs, self.prefixes)
        else:
            bNodePredicate()            
        if obj[0] == XMLLITERAL:
            # XML literal
            dom = obj[1]
            self._xwr.passXML(''.join([Canonicalize(x) for x in dom.childNodes]))
        else:
            self._xwr.data(v)
        self._xwr.endElement()

# Below is for writing an anonymous node which is the object of only one arc
# This is the arc leading to it.

# As object

    def startAnonymous(self,  tuple):
        self.startWithParseType("Resource", tuple)

    def endAnonymous(self, subject, verb):    # Remind me where we are
        self._xwr.endElement()
        self._subj = subject       # @@@ This all needs to be thought about!


# Below we do anonymous top level node - arrows only leave this circle
# There is no list synmtax for subject in RDF

# As subject

    def startAnonymousNode(self, subj, isList=0):
#        progress("@@@ Anonymous node is ", subj)
        self.flushStart()
        if self._subj:
            self._xwr.endElement()
            self._subj = None

        self._xwr.startElement(RDF_NS_URI+'Description', [], self.prefixes)
        self._subj = subj    # The object is not the subject context

    def endAnonymousNode(self, subj=None):    # Remove context
        return #If a new subject is started, they'll know to close this one
        self._xwr.endElement()
        self._subj = None
        
#  LISTS
#

# Below is for writing an anonymous node which is the object of only one arc
# This is the arc leading to it.

# As object

    def startListObject(self,  tuple, isList =0):
        self._pred = RDF_li
        self.startWithParseType('Collection', tuple)
        self.stack.append(1)
        return


    def endListObject(self, subject, verb):    # Remind me where we are
        self.stack.pop()
        self._xwr.endElement()
        self._subj = subject       # @@@ This all needs to be thought about!
        return


    def startWithParseType(self, parseType, tuple):
        self.flushStart()
        context, pred, subj, obj = tuple 
        if self._subj != subj:
            if self._subj:
                self._xwr.endElement()
            nid = self._nodeID.get(subj, None)
            if nid == None:
                subjn = self.referenceTo( subj[1])
                attr = ((RDF_NS_URI+' about', subjn),)
                if subj[0] == ANONYMOUS: attr = []
                self._xwr.startElement(RDF_NS_URI + 'Description',
                                    attr, self.prefixes)
            else:
                self._xwr.startElement(RDF_NS_URI + 'Description',
                                    ((RDF_NS_URI+' nodeID', nid),), self.prefixes)
            self._subj = subj
        nid = self._nodeID.get(pred, None)
        if nid is None:
            self._xwr.startElement(pred[1],
                    [(RDF_NS_URI+' parseType',parseType)], self.prefixes)
        else:
            bNodePredicate()             

        self._subj = obj    # The object is now the current subject
        return

# As subject:

    def startListSubject(self, subj):
        self.startAnonymousNode(self)
        # @@@@@@ set flaf to do first/rest decomp
        
    def endListSubject(self, subj):
        self.endAnonymousNode(subj)

# Below we notate a nested formula

    def startFormulaSubject(self, context):  # Doesn't work with RDF sorry ILLEGAL
        self.flushStart()
        if self._subj:
            self._xwr.endElement()
            self._subj = None
        self._xwr.startElement(RDF_NS_URI+'Description', 
                              [],
                              self.prefixes)
        
        self._xwr.startElement(NODE_MERGE_URI,
                    [(RDF_NS_URI+' parseType', "Quote")], self.prefixes)
        self._subj = None


    def endFormulaSubject(self, subj):    # Remove context
        if self._subj:
            self._xwr.endElement()   # End description if any
            self._subj = 0
        self._xwr.endElement()     # End quote
        self._subj = subj

    def startFormulaObject(self, tuple):
        self.flushStart()
        context, pred, subj, obj = tuple 
        if self._subj != subj:
            if self._subj:
                self._xwr.endElement()
            nid = self._nodeID.get(subj, None)
            if nid == None:
                progress("@@@Start anonymous node but not nodeID?", subj)
                subjn = self.referenceTo( subj[1])
                self._xwr.startElement(RDF_NS_URI + 'Description',
                                    ((RDF_NS_URI+' about', subjn),), self.prefixes)
            else:
                self._xwr.startElement(RDF_NS_URI + 'Description',
                                    ((RDF_NS_URI+' nodeID', nid),), self.prefixes)
            self._subj = subj

#        log_quote = self.prefixes[(SYMBOL, Logic_NS)] + ":Quote"  # Qname yuk
        self._xwr.startElement(pred[1], [(RDF_NS_URI+' parseType', "Quote")],
                                self.prefixes)  # @@? Parsetype RDF? Formula?
        self._subj = None


    def endFormulaObject(self, pred, subj):    # Remove context
        if self._subj:
            self._xwr.endElement()        #  </description> if any
            self._subj = None
        self._xwr.endElement()           # end quote
        self._subj = subj   # restore context from start
#       print "Ending formula, pred=", pred, "\n   subj=", subj
#        print "\nEnd bag object, pred=", `pred`[-12:]

            
    
########################################### XML Writer

class XMLWriter:
    """ taken from
    Id: tsv2xml.py,v 1.1 2000/10/02 19:41:02 connolly Exp connolly
    
    Takes as argument a writer which does the (eg utf-8) encoding
    """

    def __init__(self, encodingWriter, counter, squeaky=0, version='1.0'):
#       self._outFp = outFp
        self._encwr = encodingWriter
        self._elts = []
        self.squeaky = squeaky  # No, not squeaky clean output
        self.tab = 4        # Number of spaces to indent per level
        self.needClose = 0  # 1 Means we need a ">" but save till later
        self.noWS = 0       # 1 Means we cant use white space for prettiness
        self.currentNS = None # @@@ Hack
        self.counter = counter
        self.version = version
        
    #@@ on __del__, close all open elements?

    _namechars = string.lowercase + string.uppercase + string.digits + '_-'


    def newline(self, howmany=1):
        if self.noWS:
            self.noWS = 0
            self.flushClose()
            return
        i = howmany
        if i<1: i=1
        self._encwr("\n\n\n\n"[:i])
        self.indent()

    def indent(self, extra=0):
        self._encwr(' ' * ((len(self._elts)+extra) * self.tab))
        self.flushClose()
        
    def closeTag(self):
        if self.squeaky:
            self.needClose =1
        else:
            self._encwr(">")
            
    def flushClose(self):
        if self.needClose:
            self._encwr(">")
            self.needClose = 0

    def passXML(self, st):
        self.flushClose()
        self._encwr(st)

    def figurePrefix(self, uriref, rawAttrs, prefixes):
        i = len(uriref)
        while i>0:
            if isXMLChar(uriref[i-1], NCNameChar): # uriref[i-1] in self._namechars:
                i = i - 1
            else:
                break
        while i<len(uriref):
            if (not isXMLChar(uriref[i], NCNameStartChar)) or (uriref[i-1] == ':' and
                                                               uriref.rfind(':', 0, i-1) < 0):
                i = i+1
            else:
                break
        else:
#            raise RuntimeError
            if self.version == '1.0':
                self.version = '1.1'
                setXMLVersion('1.1')
                return self.figurePrefix(uriref, rawAttrs, prefixes)
            raise RuntimeError("this graph cannot be serialized in RDF/XML")
        # We now have the largest possible namespace. Maybe a smaller one is defined already?
        j = i
        while j<len(uriref):
            if (not isXMLChar(uriref[i], NCNameStartChar)) or uriref[i-1] == ':' or (uriref[:j] not in prefixes):
                j = j + 1
            else:
                i = j
                break
        ln = uriref[i:]
        #progress(ln)
        ns = uriref[:i]
        self.counter.countNamespace(ns)
#        print "@@@ ns=",`ns`, "@@@ prefixes =", `prefixes`
        nsSet = False
        prefix = prefixes.get(ns, ":::")
        attrs = []
        for a, v in rawAttrs:   # Caller can set default namespace
            if a == "xmlns":
                self.currentNS = v
                nsSet = True
        if ns != self.currentNS:
            if prefix == ":::" or not prefix:  # Can't trust stored null prefix
                if nsSet:
                    prefix = findLegal(prefixes, ns)
                    prefixes[ns] = prefix
                    attrs.append(('xmlns:' + prefix, ns))
                    ln = prefix + ":" + ln
                else:
                    attrs = [('xmlns', ns)]
                    self.currentNS = ns
            else:
                if prefix: ln = prefix + ":" + ln
        for at, val in rawAttrs:
            i = string.find(at," ")  #  USe space as delim like parser
            if i<=0:            # No namespace - that is fine for rdf syntax
#                print  ("# Warning: %s has no namespace on attr %s" %
#                        (ln, at)) 
                attrs.append((at, val))
                continue
            ans = at[:i]
            lan = at[i+1:]
            if ans == XML_NS_URI: prefix = "xml"
            else:
                self.counter.countNamespace(ans)
                prefix = prefixes.get(ans,":::")
                if prefix == ":::":
                    #print "finding prefix for '%s'" % ans
                    prefix = findLegal(prefixes, ans)
                    #print "--found '%s'" % prefix
                    attrs.append(( 'xmlns:' + prefix, ans))
                    prefixes[ans] = prefix
##                  raise RuntimeError("#@@@@@ tag %s: atr %s has no prefix :-( in prefix table:\n%s" %
##                      (uriref, at, `prefixes`))
            if prefix:
                attrs.append(( prefix+":"+lan, val))
            else:
                attrs.append(( lan, val))

        self.newline(3-len(self._elts))    # Newlines separate higher levels
        self._encwr("<%s" % (ln,))

        needNL = 0
        for n, v in attrs:
            if needNL:
                self.newline()
                self._encwr("   ")
            self._encwr(" %s=\"" % (n, ))
            if type(v) is type((1,1)):
#               progress("@@@@@@ toXML.py 382: ", `v`)
                v = `v`
            xmldata(self._encwr, v, self.attrEsc)
            self._encwr("\"")
            needNL = 1

            
        return (ln, attrs)

    def makeComment(self, str):
        self.newline()
        self._encwr("<!-- " + str + "-->") # @@

    def makePI(self, str):
        self._encwr('<?' + str + '?>')
        
    def startElement(self, n, attrs = [], prefixes={}):
        oldNS = self.currentNS
        ln, at2 = self.figurePrefix(n, attrs, prefixes)
        
        self._elts.append((ln, oldNS))
        self.closeTag()

    def emptyElement(self, n, attrs=[], prefixes={}):
        oldNS = self.currentNS
        ln, at2 = self.figurePrefix(n, attrs, prefixes)

        self.currentNS = oldNS  # Forget change - no nesting
        self._encwr("/")
        self.closeTag()

    def endElement(self):

        n, self.currentNS = self._elts.pop()
        self.newline()
        self._encwr("</%s" % n)
        self.closeTag()


    dataEsc = re.compile(r"[\r<>&]")  # timbl removed \n as can be in data
    attrEsc = re.compile(r"[\r<>&'\"\n]")

    def data(self, str):
        #@@ throw an exception if the element stack is empty
#       o = self._outFp
        self.flushClose()
#        xmldata(o.write, str, self.dataEsc)
        xmldata(self._encwr, str, self.dataEsc)

        self.noWS = 1  # Suppress whitespace - we are in data

    def endDocument(self):
        while len(self._elts) > 0:
            self.endElement()
        self.flushClose()
        self._encwr("\n")


##NOTHING   = -1
##SUBJECT   = 0
##PREDICATE = 1
##OBJECT    = 2
##
##FORMULA = 0
##LIST    = 1
##ANONYMOUS = 2
##
##NO = 0
##STALE = 1
##FRESH = 2
import triple_maker
tm = triple_maker

def swap(List, a, b):
    q = List[a]
    List[a] = List[b]
    List[b] = q

def findLegal(dict, str):
    ns = Set(dict.values())
    s = u''
    k = len(str)
    while k and str[k - 1] not in string.ascii_letters:
        k = k - 1
    i = k
    while i:
        if str[i - 1] not in string.ascii_letters:
            break
        i = i - 1
    j = i
#    raise RuntimeError(str[:i]+'[i]'+str[i:k]+'[k]'+str[k:])
    while j < k and (str[j:k] in ns or str[j:k][:3] == 'xml'):
        j = j + 1
    if j == k:
        # we need to find a better string
        s = str[i:k]
        while s in ns or s[:3] == 'xml':
            s = choice(string.ascii_letters) + s[:-1]
        return s
    else:
        return str[j:k]

class tmToRDF(RDFSink.RDFStructuredOutput):
    """Trying to do the same as above, using the TripleMaker interface


    """
    def __init__(self, outFp, thisURI=None, base=None, flags=""):
        RDFSink.RDFSink.__init__(self)
        if outFp == None:
            self._xwr = XMLWriter(dummyWrite, self)
        else:
            dummyEnc, dummyDec, dummyReader, encWriter = codecs.lookup('utf-8')
            z = encWriter(outFp)
            zw = z.write
            self._xwr = XMLWriter(zw, self)
        self._subj = None
        self._base = base
        self._formula = None   # Where do we get this from? The outermost formula
        if base == None: self._base = thisURI
        self._thisDoc = thisURI
        self._flags = flags
        self._nodeID = {}
        self._nextNodeID = 0
        self.namedAnonID = 0
        self._docOpen = 0  # Delay doc open <rdf:RDF .. till after binds
        def doNothing():
            pass
        self._toDo = doNothing

        def dummyClone(self):
            "retun a version of myself which will only count occurrences"
            return tmToRDF(None, self._thisDoc, base=self._base, flags=self._flags )
        
    def start(self):
        self._parts = [tm.NOTHING]
        self._triples = [[None, None, None]]
        self._classes = [None]
        self.lists = []
        self._modes = [tm.FORMULA]
        self.bNodes = []
        self._predIsOfs = [tm.NO]
        self._pathModes = [False]
        if not self._docOpen:
            if getXMLVersion() != '1.0':
                self._xwr.makePI('xml version="%s"' % getXMLVersion())
            self._toDo()
            if self.prefixes.get(RDF_NS_URI, ":::") == ":::":
                if self.namespaces.get("rdf", ":::") ==":::":
                    self.bind("rdf", RDF_NS_URI)
#            if self.prefixes.get(Logic_NS, ":::") == ":::":
#                if self.namespaces.get("log", ":::") ==":::":
#                    self.bind("log", Logic_NS)
            ats = []
            ps = self.prefixes.values()
            ps.sort()    # Cannonicalize output somewhat
            if self.defaultNamespace and "d" not in self._flags:
                if "z" in self._flags:
                    ats.append(('xmlns',
                        self.referenceTo(self.defaultNamespace)))
                else:
                    ats.append(('xmlns',self.defaultNamespace))
            for pfx in ps:
                nsvalue = self.namespaces[pfx]
                if "z" in self._flags:
                    nsvalue = self.referenceTo( nsvalue)
                ats.append(('xmlns:'+pfx, nsvalue))

            self._xwr.startElement(RDF_NS_URI+'RDF', ats, self.prefixes)
            self._nextId = 0
            self._docOpen = 1

#        self._subjs
#        self.store.startDoc()

    def end(self):
        assert len(self.lists) == 0
        #assert len(self.bNodes) == 0
        self._closeSubject()
        self._xwr.endElement()
        self._docOpen = 0
        self._xwr.endDocument()

    def referenceTo(self, uri):
        "Conditional relative URI"
        if "r" in self._flags or self._base == None:
            return uri
        return refTo(self._base, uri)


    def addNode(self, node, nameLess = 0):
        if self._modes[-1] == tm.ANONYMOUS and node is not None and self._parts[-1] == tm.NOTHING:
            raise ValueError('You put a dot in a bNode:' + `node`)
        if self._modes[-1] == tm.FORMULA or self._modes[-1] == tm.ANONYMOUS:
            self._parts[-1] = self._parts[-1] + 1
            if self._parts[-1] > 3:
                raise ValueError('Too many parts in statement: Try ending the statement')
            if node is not None:
                #print node, '+', self._triples, '++', self._parts
                if self._parts[-1] == tm.PREDICATE and self._predIsOfs[-1] == tm.STALE:
                    self._predIsOfs[-1] = tm.NO
                if self._parts[-1] == tm.SUBJECT:
                    subj = self._triples[-1][tm.SUBJECT]
                    if subj is not None and node !=  subj:
                        self._closeSubject()
                elif self._parts[-1] == tm.PREDICATE:
                    if node == (SYMBOL, RDF_type_URI) and self._classes[-1] is None:
                        #special thing for
                        self._classes[-1] = "Wait"
                    else:
                        if self._classes[-1] is None:
                            self._openSubject(self._triples[-1][tm.SUBJECT])
                            self._classes[-1] = "Never"
                        #self._openPredicate(node)
                else:
                    if self._classes[-1] == "Wait":
                        self._classes[-1] = node
                        self._openSubject(self._triples[-1][tm.SUBJECT])
                    else:
                        if node[0] == LITERAL:
                            self._openPredicate(self._triples[-1][tm.PREDICATE], args=(node[2],node[3]))
                            self._writeLiteral(node)
                            self._closePredicate()
                        elif node[0] == SYMBOL or node[0] == ANONYMOUS:
                            self._openPredicate(self._triples[-1][tm.PREDICATE], obj=node)
                        
                if nameLess:
                    node = "NameLess"
                try:
                    self._triples[-1][self._parts[-1]] = node
                except:
                    print self._parts, " - ", self._triples
                    raise
        if self._modes[-1] == tm.ANONYMOUS and self._pathModes[-1] == True:
            self.endStatement()
            self.endAnonymous()
        if self._modes[-1] == tm.LIST and nameLess == 0:
            self._modes[-1] = tm.FORMULA
            self.beginAnonymous()
            self.lists[-1] += 1
            self._modes[-1] = tm.ANONYMOUS
            self.addSymbol(List_NS + "first")
            self.addNode(node)
            self.endStatement()
            self.addNode(None)
            self.addSymbol(List_NS + "rest")
            self._modes[-1] = tm.LIST

    def nodeIDize(self, argument):
        q = argument[1]
        if (q[0] == ANONYMOUS or q[0] == SYMBOL) and q[1] in self._nodeID:
            #print '---', (RDF_NS_URI+' nodeID', self._nodeID[q[1]])
            return (RDF_NS_URI+' nodeID', self._nodeID[q[1]])
        #print '+++', (argument[0], q[1])
        return (argument[0], self.referenceTo(q[1]))
        
    def _openSubject(self, subject):
        if subject != "NameLess":
            subj = subject[1]
            q = (self.nodeIDize((RDF_NS_URI+' about', subject)),)
        else:
            q = []
        if self._classes[-1] is None:
            self._xwr.startElement(RDF_NS_URI + 'Description',
                                    q, self.prefixes)
            self._classes[-1] = subject
        else:
            self._xwr.startElement(self._classes[-1][1],
                                    q, self.prefixes)

    def _closeSubject(self):
        self._xwr.endElement()
        self._classes[-1] = None

    def _openPredicate(self, pred, args=None, obj=None, resource=None):
        if obj is None:
            if resource is not None:
                self._xwr.startElement(pred[1],  [(RDF_NS_URI+' parseType','Resource')], self.prefixes)
            else:
                attrs = []
                if args is not None:
                    dt, lang = args
                    if dt is not None:
                        attrs.append((RDF_NS_URI+' datatype', dt))
                    if lang is not None:
                        attrs.append((XML_NS_URI+' lang', lang))
                #print pred[1], attrs
                self._xwr.startElement(pred[1], attrs, self.prefixes)
        else:
            #print '++', pred[1], ' ++'
            self._xwr.emptyElement(pred[1], [self.nodeIDize((RDF_NS_URI+' resource', obj))], self.prefixes)

    def _closePredicate(self):
        self._xwr.endElement()

    def _writeLiteral(self, lit):
        self._xwr.data(lit[1])

    def IsOf(self):
        self._predIsOfs[-1] = tm.FRESH

    def checkIsOf(self):
        return self._predIsOfs[-1]

    def forewardPath(self):
        if self._modes[-1] == tm.LIST:
            a = self.lists[-1].pop()
        else:
            a = self._triples[-1][self._parts[-1]]
            self._parts[-1] = self._parts[-1] - 1
        self.beginAnonymous()
        self.addNode(a)
        self._predIsOfs[-1] = tm.FRESH
        self._pathModes[-1] = True
        
    def backwardPath(self):
        a = self._triples[-1][self._parts[-1]]
        self._parts[-1] = self._parts[-1] - 1
        self.beginAnonymous()
        self.addNode(a)
        self._pathModes[-1] = True

    def endStatement(self):
        if self._parts[-1] == tm.SUBJECT:
            pass
        elif False:
            
            if self._parts[-1] != tm.OBJECT:
                raise ValueError('try adding more to the statement' + `self._triples`)

            if self._pathModes[-1]:
                swap(self._triples[-1], tm.PREDICATE, tm.OBJECT)
            if self._predIsOfs[-1]:
                swap(self._triples[-1], tm.SUBJECT, tm.OBJECT)
            subj, pred, obj = self._triples[-1]

            if subj == '@this':
                if pred == self.forSome:
                    formula.declareExistential(obj)
                elif pred == self.forAll:
                    formula.declareUniversal(obj)
                else:
                    raise ValueError("Internal error: unknown quntifier with '@this'")
            else:
                formula.add(subj, pred, obj)
        self._parts[-1] = tm.NOTHING
        if self._modes[-1] == tm.ANONYMOUS and self._pathModes[-1]:
            self._parts[-1] = tm.SUBJECT

    def addLiteral(self, lit, dt=None, lang=None):
        a = (LITERAL, lit, dt, lang)
        self.addNode(a)

    def addSymbol(self, sym):
        #print '////', sym
        a = (SYMBOL, sym)
        self.addNode(a)
    
    def beginFormula(self):
        raise ValueError("The XML serializer does not support nested graphs.")

    def endFormula(self):
        raise ValueError("The XML serializer does not support nested graphs.")

    def beginList(self):
        self.lists.append(0)
        self._modes.append(tm.LIST)

    def endList(self):
        self._modes[-1] = tm.ANONYMOUS
        self.addSymbol(RDF_NS_URI + 'nil')
        a = self.lists.pop()
        for x in range(a):
            if self._parts[-1] != tm.PREDICATE:
                self._modes[-1] = tm.ANONYMOUS
            self.endAnonymous()
        self._modes.pop()
        print '_______________', self._modes

    def addAnonymous(self, Id):
        #print '\\\\\\\\', Id
        a = (ANONYMOUS, Id)
        if Id not in self._nodeID:
            self._nodeID[Id] = 'b'+`self._nextNodeID`
        self._nextNodeID += 1
        self.addNode(a)
        

    def beginAnonymous(self):
        self.namedAnonID += 1
        a = (ANONYMOUS, `self.namedAnonID`)
        if self._parts[-1] == tm.NOTHING:
            self.addNode(a, nameLess = 1)
        elif self._parts[-1] == tm.PREDICATE:
            self.bNodes.append(a)
            self._openPredicate(self._triples[-1][tm.PREDICATE], resource=[])
            self._modes.append(tm.ANONYMOUS)
            self._triples.append([a, None, None])
            self._parts.append(tm.SUBJECT)
            self._predIsOfs.append(tm.NO)
            self._pathModes.append(False)
        else:
            self.addAnonymous(self.namedAnonID)
        

    def endAnonymous(self):
        if self._modes[-1] != tm.ANONYMOUS:
            self._parts[-1] = tm.SUBJECT
            #self.endStatement()
            return
        if self._parts[-1] != tm.NOTHING:
            self.endStatement()
        a = self.bNodes.pop()
        self._modes.pop()
        self._triples.pop()
        self._parts.pop()
        self._predIsOfs.pop()
        self._pathModes.pop()
        self._closePredicate()

    def declareExistential(self, sym):
        self.nodeID[(SYMBOL, sym)] = _nextNodeID
        _nextNodeID += 1
        return
        formula = self.formulas[-1]
        a = formula.newSymbol(sym)
        formula.declareExistential(a)

    def declareUniversal(self, sym):
        return
        formula = self.formulas[-1]
        a = formula.newSymbol(sym)
        formula.declareUniversal(a)

    def addQuestionMarkedSymbol(self, sym):
        return
        formula = self.formulas[-2]
        a = formula.newSymbol(sym)
        formula.declareUniversal(a)
        self.addNode(a)

        
def xmldata(write, str, markupChars):
    i = 0

    while i < len(str):
        m = markupChars.search(str, i)
        if not m:
            t = str[i:]
#           for ch in str[i:]: progress( "Char ", ord(ch))
#           progress("Writer is %s" %(`write`))
#           progress("t = "+t.encode(u)
            write(t)
            break
        j = m.start()
        write(str[i:j])
        write("&#%d;" % (ord(str[j]),))
        i = j + 1
    
def bNodePredicate():
    raise ValueError("""Serialization Error:
It is not possible in RDF/XML to have a bNode in a predicate.
See <http://www.w3.org/TR/rdf-syntax-grammar/#section-Syntax-parsetype-resource>
Try using n3 instead.""")


#ends
