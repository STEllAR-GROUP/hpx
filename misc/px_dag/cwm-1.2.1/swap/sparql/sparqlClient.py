""" SPARQL Client Query for cwm architecture

"""

QL_NS = "http://www.w3.org/2004/ql#"
from sparql2cwm import SPARQL_NS
from swap.term import Env
from RDFSink import CONTEXT, PRED, SUBJ, OBJ, PARTS, ALL4
from RDFSink import  LITERAL, XMLLITERAL, LITERAL_DT, LITERAL_LANG, ANONYMOUS, SYMBOL
from xmlC14n import Canonicalize
import urllib


#from set_importer import Set, ImmutableSet


#from OrderedSequence import merge, intersection, minus, indentString

import diag
from diag import chatty_flag, tracking, progress

def makeVarCounter(string):
    counter = [0]
    def makeVar():
        counter[0] += 1
        return string + str(counter[0])
    return makeVar


makeVar = makeVarCounter('?x')
makeExistential = makeVarCounter('_:a')



def SparqlQuery(query, items, serviceURI):
    """Perform remote query as client on remote store.
        See $SWAP/query.py
    """
##    diag.chatty_flag = 99    # @@@@@@
    if diag.chatty_flag > 10:
        progress("SPARQL Query on service %s,\n\tvariables: %s;\n\texistentials: %s" %
                            (serviceURI, query.variables, query.existentials()))
        for item in items:
            progress("\tSparql query line: %s" % (`item`))
    
#    s = query.n3String()
#    progress("QUERY IS ", s)

## To make it easy to map the variables used to the ones
## we get back, use strings, and go both ways.
    vars = {}
    reverseVars = {}
    for var in query.variables:
        varString = makeVar()
        vars[var] = varString
        reverseVars[varString] = var

    patterns = []
    for item in items:
        p = []
        for part in SUBJ, PRED, OBJ:
            term = item[part]
            p.append(vars.get(term, representationOf(term)))
        patterns.append(' '.join([x for x in p]))

    queryString = "SELECT * WHERE { %s }" % ('. '.join(patterns) )
    url = serviceURI + '?' + urllib.urlencode({'query': queryString})

    response = urllib.urlopen(url)
    results = parseSparqlResults(query.store, response)
    nbs = []
    for binding in results:
        newBinding = Env()
        for key, val in binding.items():
            var = reverseVars['?' + key]
            newBinding = newBinding.bind(var, (val, newBinding.id))
        nbs.append((newBinding, None))
#    raise NotImplementedError(results)
    if diag.chatty_flag > 50:
        progress('remote query done, nbs=%s' % nbs)
    return nbs   # No bindings for testing




from xml.sax import make_parser
from xml.sax.saxutils import handler, quoteattr, escape
from xml.sax.handler import ErrorHandler

## Lots of SAX magic follows
def parseSparqlResults(store, resultString):
    parser = make_parser()
    # Workaround for bug in expatreader.py. Needed when
    # expatreader is trying to guess a prefix.
    parser.start_namespace_decl("xml", "http://www.w3.org/XML/1998/namespace")
    parser.setFeature(handler.feature_namespaces, 1)
    sparqlResults = SparqlResultsHandler(store)
    parser.setContentHandler(sparqlResults)
    parser.setErrorHandler(ErrorHandler())
    sparqlResults.setDocumentLocator(parser)
    parser.parse(resultString)
    return sparqlResults.results


RESULTS_NS = 'http://www.w3.org/2005/sparql-results#'

class BadSyntax(SyntaxError):
    pass

class SparqlResultsHandler(handler.ContentHandler):

    states = ('sparql', 'start', 'head', 'var', 'afterHead', 'results', 'result', 'binding')
    
    def __init__(self, store):
        self.store = store


    def setDocumentLocator(self, locator):
        self.locator = locator

    def onError(self, explanation):
        documentLocation = "%s:line:%s, column:%s" % (self.locator.getSystemId(),
                            self.locator.getLineNumber(), self.locator.getColumnNumber())
        raise BadSyntax(explanation + '\n' + documentLocation)

    def startDocument(self):
#        progress('starting document')
        self.state = 'start'

    def endDocument(self):
#        progress('ending document')
        pass
#        raise NotImplementedError

    def startElementNS(self, name, qname, attrs):
        self.text = u''
        (ns, lname) = name
        if ns != RESULTS_NS:
            self.onError('The tag %s does not belong anywhere!' % (ns + lname))
        try:
            processor = self.tagStateStartHandlers[(self.state, lname)]
        except KeyError:
            self.onError("The tag %s does not belong here\nI'm in state %s" % (ns + lname, self.state))
        processor(self, attrs)
        self.text = ''

    def endElementNS(self, name, qname):
        (ns, lname) = name
        processor = self.tagEndHandlers[lname]
        processor(self)

    def characters(self, content):
        self.text = content

    def startSparql(self, attrs):
        self.state = 'sparql'
    def sparqlHead(self, attrs):
        self.state = 'head'
    def variable(self, attrs):
        self.state = 'var'
#        progress('I need to declare %s' % dict(attrs))
    def results(self, attrs):
        self.state = 'results'
        self.results = []
    def result(self, attrs):
        self.state = 'result'
        self.result = {}
    def binding(self, attrs):
        try:
            self.varName = attrs[(None, 'name')]
        except KeyError:
            self.error('We need a name')
        self.state = 'binding'
    def uri(self, attrs):
        self.state = 'uri'
    def bnode(self, attrs):
        self.state = 'bnode'
    def literal(self, attrs):
        self.state = 'literal'
 #       progress('The attrs are %s' % dict(attrs))
        self.dt = attrs.get((None, 'datatype'), None)
        if self.dt is not None:
            self.dt = self.store.newSymbol(self.dt)
        self.lang = attrs.get(('http://www.w3.org/XML/1998/namespace', 'lang'), None)

    def boolean(self, attrs):
        self.state = 'boolean'
    
    

    tagStateStartHandlers = \
                     {('start', 'sparql'): startSparql,
                      ('sparql', 'head'): sparqlHead,
                      ('head', 'variable'): variable,
                      ('afterHead', 'results'): results,
                      ('afterHead', 'boolean'): boolean,
                      ('results', 'result'): result,
                      ('result', 'binding'): binding,
                      ('binding', 'uri'): uri,
                      ('binding', 'bnode'): bnode,
                      ('binding', 'literal'): literal}

    def endHead(self):
        self.state = 'afterHead'
    def endVar(self):
        self.state = 'head'
    def endResults(self):
        self.state = 'sparql'
    def endResult(self):
        self.results.append(self.result)
        self.state = 'results'
    def endBinding(self):
        self.result[self.varName] = self.val
        self.state = 'result'
    def endLiteral(self):
        self.state = 'endBinding'
        self.val = self.store.newLiteral(self.text, self.dt, self.lang)
    def endUri(self):
        self.state = 'endBinding'
        self.val = self.store.newSymbol(self.text)
    def endBnode(self):
        self.state = 'endBinding'
        self.val = makeExistential()
    def endBoolean(self):
        self.results = (self.text == 'true')
    
    tagEndHandlers = \
                        {'sparql': lambda x: None,
                         'head': endHead,
                         'variable': endVar,
                         'results': endResults,
                         'result': endResult,
                         'binding': endBinding,
                         'literal': endLiteral,
                         'uri': endUri,
                         'bnode': endBnode,
                         'boolean': endBoolean}

from notation3 import stringToN3, backslashUify, N3_nil
from pretty import auPair

def representationOf(pair):
    """  Representation of a thing in the output stream

    Regenerates genids if required.
    Uses prefix dictionary to use qname syntax if possible.
    """
    pair = auPair(pair)
    _flags = ''

    if "t" not in _flags:
        if pair == N3_nil:
            return"()"

    ty, value = pair

    singleLine = False
    if ty == LITERAL:
        return stringToN3(value, singleLine=singleLine, flags = _flags)

    if ty == XMLLITERAL:
        st = Canonicalize(value, None, unsuppressedPrefixes=['foo'])
        st = stringToN3(st, singleLine=singleLine, flags=_flags)
        return st + "^^" + representationOf((SYMBOL,
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral"))

    if ty == LITERAL_DT:
        s, dt = value
        if "b" not in _flags:
            if (dt == BOOLEAN_DATATYPE):
                return toBool(s) and "true" or "false"
        if "n" not in _flags:
            dt_uri = dt
#               dt_uri = dt.uriref()             
            if (dt_uri == INTEGER_DATATYPE):
                return str(long(s))
            if (dt_uri == FLOAT_DATATYPE):
                retVal =  str(float(s))    # numeric value python-normalized
                if 'e' not in retVal:
                    retVal += 'e+00'
                return retVal
            if (dt_uri == DECIMAL_DATATYPE):
                retVal = str(Decimal(s))
                if '.' not in retVal:
                    retVal += '.0'
                return retVal
        st = stringToN3(s, singleLine= singleLine, flags=_flags)
        return st + "^^" + representationOf((SYMBOL, dt))

    if ty == LITERAL_LANG:
        s, lang = value
        return stringToN3(s, singleLine= singleLine,
                                    flags=_flags)+ "@" + lang

#    aid = self._anodeId.get(pair[1], None)
#    if aid != None:  # "a" flag only
#        return "_:" + aid    # Must start with alpha as per NTriples spec.

##    if ((ty == ANONYMOUS)
##        and not option_noregen and "i" not in self._flags ):
##            x = self.regen.get(value, None)
##            if x == None:
##                x = self.genId()
##                self.regen[value] = x
##            value = x
###                return "<"+x+">"
##
##
##    j = string.rfind(value, "#")
##    if j<0 and "/" in self._flags:
##        j=string.rfind(value, "/")   # Allow "/" namespaces as a second best
##    
##    if (j>=0
##        and "p" not in self._flags):   # Suppress use of prefixes?
##        for ch in value[j+1:]:  #  Examples: "." ";"  we can't have in qname
##            if ch in _notNameChars:
##                if verbosity() > 0:
##                    progress("Cannot have character %i in local name."
##                                % ord(ch))
##                break
##        else:
##            namesp = value[:j+1]
##            if (self.defaultNamespace
##                and self.defaultNamespace == namesp
##                and "d" not in self._flags):
##                return ":"+value[j+1:]
##            self.countNamespace(namesp)
##            prefix = self.prefixes.get(namesp, None) # @@ #CONVENTION
##            if prefix != None : return prefix + ":" + value[j+1:]
##        
##            if value[:j] == self.base:   # If local to output stream,
##                return "<#" + value[j+1:] + ">" # use local frag id
    
##    if "r" not in self._flags and self.base != None:
##        value = hexify(refTo(self.base, value))
##    elif "u" in self._flags:
    value = backslashUify(value)
##    else: value = hexify(value)

    return "<" + value + ">"    # Everything else

