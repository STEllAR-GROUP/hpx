"""Why??

$$
"""



from n3p import n3p
import os
from uripath import join

class absolutizer(unicode):
    def __getitem__(self, other):
        return join(self, other)

Logic_NS = "http://www.w3.org/2000/10/swap/log#"
RDF_type_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDF_NS_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
# DAML_NS=DPO_NS = "http://www.daml.org/2001/03/daml+oil#"  # DAML plus oil
OWL_NS = "http://www.w3.org/2002/07/owl#"
DAML_sameAs_URI = OWL_NS+"sameAs"
parsesTo_URI = Logic_NS + "parsesTo"
RDF_spec = "http://www.w3.org/TR/REC-rdf-syntax/"

#List_NS = DPO_NS     # We have to pick just one all the time
List_NS = RDF_NS_URI     # From 20030808


LOG_implies_URI = "http://www.w3.org/2000/10/swap/log#implies"

INTEGER_DATATYPE = "http://www.w3.org/2001/XMLSchema#integer"
FLOAT_DATATYPE = "http://www.w3.org/2001/XMLSchema#double"
DECIMAL_DATATYPE = "http://www.w3.org/2001/XMLSchema#decimal"

NORMAL = 1
PREFIX = 2
FORSOME = 4
FORALL = 5
NEED_PERIOD = 8
LITERAL = 9
KEYWORDS = 10

class n3p_tm(n3p.N3Parser):
    def __init__(self, uri, tm, baseURI=False):
        n3p.N3Parser.__init__(self, 'nowhere', n3p.branches, n3p.regexps)
        self.tm = tm
        self.productions = []
        if baseURI:
            self._baseURI = baseURI + '#'
        else:
            self._baseURI = uri + '#'
        self.prefixes = {'':self._baseURI}
        self.abs = absolutizer(self._baseURI[:-1])

    def parse(self, start=n3p.start):
        n3p.N3Parser.parse(self, start)

    def startDoc(self):
        self.tm.start()

    def endDoc(self):
        return self.tm.end()

    def feed(self, data):
        self.data = data.decode('utf_8')
        self._mode = NORMAL
        self.parse()

    def onStart(self, prod):
        self.productions.append(prod)
        if prod == 'literal':
            self._mode = LITERAL
        #print 'begin' + prod

    def onFinish(self):
        prod = self.productions.pop()
        if prod == 'literal':
            self.tm.addLiteral(self.literal['string'], self.literal['type'], self.literal['lang'])
            self._mode = NORMAL
        #print 'end' + prod

    def onToken(self, prod, tok):
        tm = self.tm
        if 1:
            if self._mode == NORMAL:
                if prod == '.':
                    try:
                        tm.endStatement()
                    except ValueError:
                        tm.forewardPath()
                elif prod == ';':
                    tm.endStatement()
                    tm.addNode(None)
                elif prod == ',':
                    tm.endStatement()
                    tm.addNode(None)
                    tm.addNode(None)
                elif prod == '=>':
                    tm.addSymbol(LOG_implies_URI)

                elif prod == '<=':
                    tm.IsOf()
                    tm.addSymbol(LOG_implies_URI)
                elif prod == '=':
                    tm.addSymbol(DAML_sameAs_URI)
                elif prod == '@forAll':
                    self._mode = FORALL
                elif prod == '@forSome':
                    self._mode = FORSOME
                elif prod == '@a':
                    tm.addSymbol(RDF_type_URI)
                elif prod == '@is':
                    tm.IsOf()
                elif prod == '@of':
                    assert tm.checkIsOf()
                elif prod == '{':
                    tm.beginFormula()
                elif prod == '}':
                    tm.endFormula()
                elif prod == '[':
                    tm.beginAnonymous()
                elif prod == ']':
                    tm.endAnonymous()
                elif prod == '(':
                    tm.beginList()
                elif prod == ')':
                    tm.endList()
                elif prod == '@has':
                    pass
                elif prod == '@this':
                    tm.addNode('@this')
                elif prod == '^':
                    tm.backwardPath()
                elif prod == '!':
                    tm.forewardPath()
                elif prod == '@prefix':
                    self._mode = PREFIX
                elif prod == 'qname':
                    ret = self.decodeQName(tok, True)
                    if type(ret) is tuple:
                        tm.addAnonymous(ret[0])
                    else:
                        tm.addSymbol(ret)
                elif prod == 'explicituri':
                    tm.addSymbol(self.abs[tok[1:-1]])
                elif prod == 'numericliteral':
                    if '.' in tok or 'e' in tok:
                        tm.addLiteral(float(tok))
                    else:
                        tm.addLiteral(int(tok))
                elif prod == 'variable':
                    tm.addQuestionMarkedSymbol(self._baseURI+tok[1:])
                elif prod == '@keywords':
                    self._mode = KEYWORDS
                
            elif self._mode == LITERAL:
                if prod == 'string':
                    self.literal = {'string': unEscape(tok), 'type':None, 'lang':None}
                elif prod == 'qname':
                    self.literal['type'] = self.decodeQName(tok, False)
                elif prod == 'explicituri':
                    self.literal['type'] = self.abs[tok[1:-1]]
                elif prod == 'langcode':
                    self.literal['lang'] = tok
                else:
                    pass
            elif self._mode == PREFIX:
                if prod == 'qname':
                    n = tok.find(':')
                    self._newPrefix = tok[:n]
                elif prod == 'explicituri':
                    tm.bind(self._newPrefix, self.abs[tok[1:-1]])
                    self.prefixes[self._newPrefix] = self.abs[tok[1:-1]]
                elif prod == '.':
                    self._mode = NORMAL
            elif self._mode == FORALL:
                if prod == 'qname':
                    uri = self.decodeQName(tok, False)
                    tm.declareUniversal(uri)
                elif prod == 'explicituri':
                    tm.declareUniversal(self.abs[tok[1:-1]])
                elif prod == ',':
                    pass
                elif prod == '.':
                    self._mode = NORMAL
            elif self._mode == FORSOME:
                if prod == 'qname':
                    uri = self.decodeQName(tok, False)
                    tm.declareExistential(uri)
                elif prod == 'explicituri':
                    tm.declareExistential(self.abs[tok[1:-1]])
                elif prod == ',':
                    pass
                elif prod == '.':
                    self._mode = NORMAL
                    
        if prod == '.':
            self._mode = NORMAL
        parent = self.productions[-1]
        #print 'mode=', self._mode, '. prod=', prod, '. tok =', tok


    def decodeQName(self, qName, bNode):
        n = qName.find(':')
        if n < 0:
            pfx = ''
            frag = qName
        else:
            pfx = qName[:n]
            frag = qName[n+1:]
        try:
            self.prefixes[pfx]
        except KeyError:
            if pfx == '_':
                if bNode:
                    return (frag,)
                else:
                    return qName
            raise
        return self.prefixes[pfx]+frag

def unEscape(string):
    if string[:3] == '"""':
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
            k = 'abfrtvn\\"'.find(a)
            if k >= 0:
                ret += '\a\b\f\r\t\v\n\\"'[k]
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



def parse(uri, options): 
   baseURI = options.baseURI
   sink = object()
   if options.root: 
      sink.quantify = lambda *args: True
      sink.flatten = lambda *args: True
   if ':' not in uri: 
      uri = 'file://' + os.path.join(os.getcwd(), uri)
   if baseURI and (':' not in baseURI): 
      baseURI = 'file://' + os.path.join(os.getcwd(), baseURI)
   p = n3p_tm(uri, sink, baseURI=baseURI)
   p.parse()

def main(argv=None): 
   import optparse

   class MyHelpFormatter(optparse.HelpFormatter): 
      def __init__(self): 
         kargs = {'indent_increment': 2, 'short_first': 1, 
                  'max_help_position': 25, 'width': None}
         optparse.HelpFormatter.__init__(self, **kargs)
      def format_usage(self, usage): 
         return optparse._("%s") % usage.lstrip()
      def format_heading(self, heading): 
         return "%*s%s:\n" % (self.current_indent, "", heading)
   formatter = MyHelpFormatter()

   parser = optparse.OptionParser(usage=__doc__, formatter=formatter)
   parser.add_option("-b", "--baseURI", dest="baseURI", default=False, 
                     help="set the baseURI", metavar="URI")
   parser.add_option("-r", "--root", dest="root", 
                     action="store_true", default=False, 
                     help="print triples in the root formula only")
   options, args = parser.parse_args(argv)

   if len(args) == 1: 
      parse(args[0], options)
   else: parser.print_help()

if __name__=="__main__": 
   main()
