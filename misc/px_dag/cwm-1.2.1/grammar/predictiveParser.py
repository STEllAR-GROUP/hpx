#! /usr/bin/python
"""Generic predictive parser for N3-like languages

This is is a generic parser for a set of N3-like languages.
It is directly driven by the context-free grammar (CFG) in RDF.
It was made to mutually test the CFG against the test files.

Options

--parse=uri   This is the document to be parsed. Optional.

--as=uri      This is the URI of the production as which the document
              is to be parsed.
              default: http://www.w3.org/2000/10/swap/grammar/n3#document

--grammar=uri This is the RDF augmented grammar.  Default is the
              as= production's URI with the hash stripped.

--yacc=file   If this is given a yacc format grammar will be generated
              and written to the given file.
              
(Abbreviated options -p, -a, -g, -y)

For example:
cwm n3.n3 bnf-rules.n3 --think --purge --data > n3-selectors.n3
PYTHONPATH=$SWAP python predictiveParser.py --grammar=n3-selectors.n3  \
    --as=http://www.w3.org/2000/10/swap/grammar/n3#document  --parse=n3.n3
    
The parser is N3-specific only because of the built-in tokenizer.
This program is or was http://www.w3.org/2000/10/swap/grammar/predictiveParser.py
W3C open source licence. Enjoy. Tim BL
"""

__version__ = "$Id: predictiveParser.py,v 1.19 2007/06/26 02:36:16 syosi Exp $"

# SWAP http://www.w3.org/2000/10/swap
try:
    from swap import webAccess, uripath, llyn, myStore, term, diag
    from swap.myStore import load, Namespace
    from swap.term import Literal
    from swap.diag import progress, chatty_flag
except ImportError:
    import webAccess, uripath, llyn, myStore, term, diag
    from myStore import load, Namespace
    from term import Literal
    from diag import progress, chatty_flag    

#diag.chatty_flag=0

# Standard python
import sys, getopt
from sys import exit, stderr
from time import clock
import re
from codecs import utf_8_encode

BNF = Namespace("http://www.w3.org/2000/10/swap/grammar/bnf#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
REGEX = Namespace("http://www.w3.org/2000/10/swap/grammar/regex#")
    

branchTable = {}
tokenRegexps = {}

def recordError(str):
    global errors
    errors.append(str)
    str2 =  "##### ERROR:  " + str
    return str2

def deColonise(s):
    t = ""
    for ch in s:
        if ch == ":": ch = "_"
        t = t + ch
    return t

def toYacc(x, tokenRegexps):
    if isinstance(x, Literal):
        return "'" + str(x.value()) + "'"  # @@@ Escaping
    if x in tokenRegexps:
        return deColonise(`x`).upper()
    return deColonise(`x`)

        
def yaccConvert(yacc, top, tokenRegexps):
    "Convert the grammar to yacc format"
    global already, agenda, errors
    already = []
    agenda = []
    errors = []
    for x in tokenRegexps:
        yacc.write("%%token %s\n" % toYacc(x, tokenRegexps))
    yacc.write("\n")
    yacc.write("%%\n")
    yaccProduction(yacc, top, tokenRegexps)
    while agenda:
        x = agenda[0]
        agenda = agenda[1:]
        already.append(x)
        yaccProduction(yacc, x, tokenRegexps)
    yacc.write("eof:  /* empty */; \n")
    yacc.write("%%\n")
    
def yaccProduction(yacc, lhs,  tokenRegexps):
    if lhs is BNF.void:
        if chatty_flag: progress( "\nvoid")
        return
    if lhs is BNF.eof:
        if chatty_flag: progress( "\nEOF")
        return
    if isinstance(lhs, Literal):
        literalTerminals[lhs.value()] = 1
#       print "\nLiteral %s" %(lhs)
        return

    rhs = g.the(pred=BNF.matches, subj=lhs)
    if rhs != None:
        if chatty_flag: progress( "\nToken %s matches regexp %s" %(lhs, rhs))
#       tokenRegexps[lhs] = re.compile(rhs.value())
        return
    rhs = g.the(pred=BNF.mustBeOneSequence, subj=lhs)
    if rhs == None:
        progress( recordError("No definition of " + `lhs`))
        raise ValueError("No definition of %s  in\n %s" %(`lhs`, `g`))
    options = rhs
    if chatty_flag:
        progress ("\nProduction %s :: %s  ie %s" %(`lhs`, `options` , `options.value()`))
    yacc.write("\n%s:" % toYacc(lhs, tokenRegexps))

    branches = g.each(subj=lhs, pred=BNF.branch)
    first = 1
    for branch in branches:
        if not first:
            yacc.write("\t|\t")
        first = 0
        option = g.the(subj=branch, pred=BNF.sequence)
        if chatty_flag: progress( "\toption: "+`option.value()`)
        yacc.write("\t")
        if option.value() == [] and yacc: yacc.write(" /* empty */")
        for part in option:
            if part not in already and part not in agenda: agenda.append(part)
            yacc.write(" %s" % toYacc(part, tokenRegexps))
        yacc.write("\n")
    yacc.write("\t;\n")
        
########################################## end of yacc converter

literalTerminals = {}
def doProduction(lhs):
    "Generate branch tables for one production"
    global branchTable
    if lhs is BNF.void:
        progress("\nvoid")
        return
    if lhs is BNF.eof:
        progress( "\nEOF")
        return
    if isinstance(lhs, Literal):
        literalTerminals[lhs.value()] = 1
        return

    branchDict = {}

    rhs = g.the(pred=BNF.matches, subj=lhs)
    if rhs != None:
        if chatty_flag: progress( "\nToken %s matches regexp %s" %(lhs, rhs))
        try:
            tokenRegexps[lhs] = re.compile(rhs.value(), re.U)
        except:
            print rhs.value().encode('utf-8')
            raise
        cc = g.each(subj=lhs, pred=BNF.canStartWith)
        if cc == []: progress (recordError(
            "No record of what token %s can start with" % `lhs`))
        if chatty_flag: progress("\tCan start with: %s" % cc) 
        return
    if g.contains(subj=lhs, pred=RDF.type, obj=REGEX.Regex):
        import regex
        rhs = regex.makeRegex(g, lhs)
        try:
            tokenRegexps[lhs] = re.compile(rhs, re.U)
        except:
            print rhs
            raise
        cc = g.each(subj=lhs, pred=BNF.canStartWith)
        if cc == []: progress (recordError(
            "No record of what token %s can start with" % `lhs`))
        if chatty_flag: progress("\tCan start with: %s" % cc) 
        return         
    
    rhs = g.the(pred=BNF.mustBeOneSequence, subj=lhs)
    if rhs == None:
        progress (recordError("No definition of " + `lhs`))
        return
#       raise RuntimeError("No definition of %s  in\n %s" %(`lhs`, `g`))
    options = rhs
    if chatty_flag: progress ( "\nProduction %s :: %s  ie %s" %(`lhs`, `options` , `options.value()`))
    succ = g.each(subj=lhs, pred=BNF.canPrecede)
    if chatty_flag: progress("\tCan precede ", succ)

    branches = g.each(subj=lhs, pred=BNF.branch)
    for branch in branches:
        option = g.the(subj=branch, pred=BNF.sequence)
        if chatty_flag: progress( "\toption: "+`option.value()`)
        for part in option:
            if part not in already and part not in agenda: agenda.append(part)
            y = `part`
        conditions = g.each(subj=branch, pred=BNF.condition)
        if conditions == []:
            progress(
                recordError(" NO SELECTOR for %s option %s ie %s" %
                (`lhs`, `option`, `option.value()` )))
            if option.value == []: # Void case - the tricky one
                succ = g.each(subj=lhs, pred=BNF.canPrecede)
                for y in succ:
                    if chatty_flag: progress("\t\t\tCan precede ", `y`)
        if chatty_flag: progress("\t\tConditions: %s" %(conditions))
        for str1 in conditions:
            if str1 in branchDict:
                progress(recordError(
                    "Conflict: %s is also the condition for %s" % (
                                str1, branchDict[str1].value())))
            branchDict[str1.__str__()] = option
#           break

    for str1 in branchDict:
        for str2 in branchDict:
            s1 = unicode(str1)
            s2 = unicode(str2)
# @@ check that selectors are distinct, not substrings
            if (s1.startswith(s2) or s2.startswith(s1)) and branchDict[str1] is not branchDict[str2]:
                progress("WARNING: for %s, %s indicates %s, but  %s indicates %s" % (
                            lhs, s1, branchDict[str1], s2, branchDict[str2]))
    branchTable[lhs] = branchDict


######################### Parser based on the RDF Context-free grammar

whiteSpace = re.compile(ur'[ \t]*((#[^\n]*)?\r?\n)?')
singleCharacterSelectors = u"\t\r\n !\"#$%&'()*.,+/;<=>?[\\]^`{|}~"
notQNameChars = singleCharacterSelectors + "@"  # Assume anything else valid qname :-/
notNameChars = notQNameChars + ":"  # Assume anything else valid name :-/

class PredictiveParser:
    """A parser for N3 or derived languages"""

    def __init__(parser, sink, top,  branchTable, tokenRegexps, keywords = None):
        parser.sink = sink
        parser.top = top    #  Initial production for whole document
        parser.branchTable = branchTable
        parser.tokenRegexps = tokenRegexps
        parser.lineNumber = 1
        parser.startOfLine = 0  # Offset in buffer
        parser.atMode = True
        if keywords:
            parser.keywords = keywords
            parser.atMode = False
        else:
            parser.keywords = [ "a", "is", "of", "this" ]
        #print parser.keywords
        parser.verb = 0  # Verbosity
        parser.keywordMode = 0  # In a keyword statement, adding keywords
        
    def countLines(parser, buffer, here):
        """Count lines since called last time
        
        Make sure we count all lines
        """
        parser.lineNumber+= buffer[parser.startOfLine:here].count("\n")
        parser.startOfLine = here

    def token(parser, str, i):
        """The Tokenizer:  returns (token type character, offset of token)
        Skips spaces.
        "0" means numeric
        "a" means alphanumeric
        """
        
        while 1:
            m = whiteSpace.match(str, i)
            if m == None or m.end() == i: break
            i = m.end()
        parser.countLines(str, i)
        if i == len(str):
            return "",  i # eof
        
        if parser.verb: progress( "%i) Looking at:  ...%s$%s..." % (
            parser.lineNumber, str[i-10:i],str[i:i+10]))
        for double in "=>", "<=", "^^":
            if double == str[i:i+2]: return double, i
    
        ch = str[i]
        if ch == ".": parser.keywordMode = 0 # hack
        if ch in singleCharacterSelectors:
            return ch, i
        if ch in "+-0123456789":
            return "0", i #  Numeric
        j = i+1
        if ch == "@":
            if i!=0 and whiteSpace.match(str[i-1]).end() == 0:
                return ch, i
            while str[j] not in notNameChars: j = j + 1
            if str[i+1:j] == "keywords" :
                parser.keywords = [] # Special
                parser.keywordMode = 1
            return str[i:j], i # keyword
        if ch == '"':  #"
            return '"', i #"
   
        # Alphanumeric: keyword hacks
        while str[j] not in notQNameChars: j = j+1
        word = str[i:j]
        if parser.keywordMode:
            parser.keywords.append(word)
        elif word in parser.keywords:
            if word == "keywords" :
                parser.keywords = []    # Special
                parser.keywordMode = 1
            if parser.atMode:
                return "@" + word, i  # implicit keyword
            return word, i
        return "a", i    # qname, langcode, or barename

    def around(parser, str, here):
        "The line around the given point"
        sol = str.rfind("\n", 0, here)
        if sol <= 0: sol = 0
        eol = str.find("\n", here)
        if eol <= 0: eol = len(str)
        return "On line %i at $ in: %s$%s" %(parser.lineNumber, str[sol:here], str[here:eol])
        
    def parse(parser, str):
        tok, here = parser.token(str, 0)
        return parser.parseProduction(parser.top, str, tok, here)
        
    def parseProduction(parser, lhs, str, tok=None, here=0):
        "The parser itself."

        if tok == "": return tok, here # EOF    
        lookupTable = parser.branchTable[lhs]
        rhs = lookupTable.get(tok, None)  # Predict branch from token
        if rhs == None:
            progress("""Found %s when expecting some form of %s,
\tsuch as %s\n\t%s"""  % (tok, lhs, lookupTable.keys(), parser.around(str, here)))
            raise SyntaxError("""Found %s when expecting some form of %s,
\tsuch as %s\n\t%s"""  % (tok, lhs, lookupTable.keys(), parser.around(str, here)))
        if parser.verb: progress( "%i  %s means expand %s as %s" %(parser.lineNumber,tok, lhs, rhs.value()))
        for term in rhs:
            if isinstance(term, Literal): # CFG Terminal
                lit = term.value()
                next = here + len(lit)
                if str[here:next] == lit: pass
                elif "@"+str[here:next-1] == lit: next = next-1
                else: raise SyntaxError(
                    "Found %s where %s expected\n\t %s" %
                        (`str[here:next]`, lit, parser.around(str, here)))
            else:
                rexp = tokenRegexps.get(term, None)
                if rexp == None: # Not token
                    tok, here = parser.parseProduction(term, str, tok, here)
                    continue
                m = rexp.match(str, here)
                if m == None:
                    progress("\n\n\nToken: should match %s\n\t %s" % 
                                (rexp.pattern, parser.around(str, here)))
                    raise SyntaxError("Token: should match %s\n\t %s" % 
                                (rexp.pattern, parser.around(str, here)))
                if parser.verb: progress( "Token matched to <%s> as pattern <%s>" % (str[here:m.end()], rexp.pattern))
                next = m.end()
            tok, here = parser.token(str, next)  # Next token
        return tok, here

###############################

def _test():
    import doctest, predictiveParser
    doctest.testmod(uripath)

def usage():
    sys.stderr.write(__doc__)

############################### Program

def main():
    global already, agenda, errors
    parseAs = None
    grammarFile = None
    parseFile = None
    yaccFile = None
    global verbose
    global g
    verbose = 0
    lumped = 1

    try:
        opts, args = getopt.getopt(sys.argv[1:], "ha:v:p:g:y:",
            ["help", "as=",  "verbose=", "parse=", "grammar=", "yacc="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    output = None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        if o in ("-v", "--verbose"):
            verbose =int(a)
            diag.chatty_flag = int(a)
        if o in ("-a", "--as"):
            parseAs = uripath.join(uripath.base(), a)
        if o in ("-p", "--parse"):
            parseFile = uripath.join(uripath.base(), a)
        if o in ("-g", "--grammar"):
            grammarFile = uripath.join(uripath.base(), a)
        if o in ("-y", "--yacc"):
            yaccFile = uripath.join(uripath.base(), a)[5:]  # strip off file:

    

#    if testFiles == []: testFiles = [ "/dev/stdin" ]
    if not parseAs:
        usage()
        sys.exit(2)
    parseAs = uripath.join(uripath.base(), parseAs)
    if not grammarFile:
        grammarFile = parseAs.split("#")[0]   # strip off fragid
    else:
        grammarFile = uripath.join(uripath.base(), grammarFile)


    
    # The Grammar formula
    progress("Loading " + grammarFile)
    start = clock()
    g = load(grammarFile)
    taken = clock() - start + 1
    progress("Loaded %i statements in %fs, ie %f/s." %
        (len(g), taken, len(g)/taken))
    
    document = g.newSymbol(parseAs)
    
    already = []
    agenda = []
    errors = []
    doProduction(document)
    while agenda:
        x = agenda[0]
        agenda = agenda[1:]
        already.append(x)
        doProduction(x)
        
    if errors != []:
        progress("###### FAILED with %i errors." % len(errors))
        for s in errors: progress ("\t%s" % s)
        exit(-2)
    else:
        progress( "Ok for predictive parsing")
    
    #if parser.verb: progress "Branch table:", branchTable
    if verbose:
        progress( "Literal terminals: %s" %  literalTerminals.keys())
        progress("Token regular expressions:")
        for r in tokenRegexps:
            progress( "\t%s matches %s" %(r, tokenRegexps[r].pattern) )
    
    if yaccFile:
        yacc=open(yaccFile, "w")
        yaccConvert(yacc, document, tokenRegexps)
        yacc.close()

    if parseFile == None: exit(0)

    
    ip = webAccess.urlopenForRDF(parseFile, None)
    
    str = ip.read().decode('utf_8')
    sink = g.newFormula()
    keywords = g.each(pred=BNF.keywords, subj=document)
    keywords = [a.value() for a in keywords]
    p = PredictiveParser(sink=sink, top=document, branchTable= branchTable,
            tokenRegexps= tokenRegexps, keywords =  keywords)
    p.verb = verbose
    start = clock()
    p.parse(str)
    taken = clock() - start + 1
    progress("Loaded %i chars in %fs, ie %f/s." %
        (len(str), taken, len(str)/taken))
    progress("Parsed <%s> OK" % parseFile)
    sys.exit(0)   # didn't crash
    
###########################################################################

        
                
if __name__ == "__main__":
    main()


#ends
