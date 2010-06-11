#!/usr/local/bin/python
"""Web Access

This module implements some basic bits of the web architecture:
dereferencing a URI to get a document, with content negotiation,
and deciding on the basis of the Internet Content Type what to do with it.

$Id: webAccess.py,v 1.34 2007/08/06 16:13:56 syosi Exp $


Web access functionality building on urllib2

"""

import sys, os

#import urllib
import urllib2, urllib  # Python standard

from why import newTopLevelFormula

import uripath # http://www.w3.org/2000/10/swap/uripath.py
import diag
from diag import progress
import notation3   # Parser    @@@ Registery of parsers vs content types woudl be better.

from OrderedSequence import indentString

HTTP_Content_Type = 'content-type' #@@ belongs elsewhere?

print_all_file_names = diag.print_all_file_names   # for listing test files

class SecurityError(IOError):
    pass

# A little code to represent a value that can be set
# and read; a singleton. In essence, this is a little
# prettier than a one element list
def setting(self, val=None):
    if val is not None:
        self[0] = val
    return self[0]

sandBoxed = setting.__get__([False])

def cacheHack(addr):
    """ If on a plane, hack remote w3.org access to local access
    """
    real = "http://www.w3.org/"
    local = "/devel/WWW/"
    suffixes = [ "", ".rdf", ".n3" ]
    if addr.startswith(real):
        rest = local + addr[len(real):]
        for s in suffixes:
            fn = rest + s
            try:
                os.stat(fn)
                progress("Offline: Using local copy %s" % fn)
                return "file://" + fn
            except OSError:
                continue
    return addr
                
def urlopenForRDF(addr, referer=None):
    """Access the web, with a preference for RDF
    """
    return webget(addr,
                  types=['text/rdf+n3',
                         'application/rdf+xml'
       #                  ,'application/x-turtle'    # Why not ask for turtle?
                         ], 
                  referer = referer)


def webget(addr, referer=None, types=[]):
    """Open a URI for reading; return a file-like object with .headers
    cf http://www.w3.org/TR/2004/REC-webarch-20041215/#dereference-uri
    """

    if diag.chatty_flag > 7: progress("Accessing: " + addr)
    if sandBoxed():
        if addr[:5] == 'file:':
            raise SecurityError('local file access prohibited')

#    addr = cacheHack(addr)

    # work around python stdlib bugs with data: URIs
    # buggy in 2.4.2 with CStringIO
    if addr[:5] == 'data:':
        # return open_data(addr)
        return urllib.urlopen(addr)

    req = urllib2.Request(addr)

    if types:
        req.add_header('Accept', ','.join(types))

    if referer: #consistently misspelt
        req.add_header('Referer', referer)

    stream =  urllib2.urlopen(req)

    if print_all_file_names:
        diag.file_list.append(addr)

    return stream


def load(store, uri=None, openFormula=None, asIfFrom=None, contentType=None,
                flags="", referer=None, why=None, topLevel=False):
    """Get and parse document.  Guesses format if necessary.

    uri:      if None, load from standard input.
    remember: if 1, store as metadata the relationship between this URI and this formula.
    
    Returns:  top-level formula of the parsed document.
    Raises:   IOError, SyntaxError, DocumentError
    
    This is an independent function, as it is fairly independent
    of the store. However, it is natural to call it as a method on the store.
    And a proliferation of APIs confuses.
    """
#    if referer is None:
#        raise RuntimeError("We are trying to force things to include a referer header")
    try:
        baseURI = uripath.base()
        if uri != None:
            addr = uripath.join(baseURI, uri) # Make abs from relative
            if diag.chatty_flag > 40: progress("Taking input from " + addr)
            netStream = urlopenForRDF(addr, referer)
            if diag.chatty_flag > 60:
                progress("   Headers for %s: %s\n" %(addr, netStream.headers.items()))
            receivedContentType = netStream.headers.get(HTTP_Content_Type, None)
        else:
            if diag.chatty_flag > 40: progress("Taking input from standard input")
            addr = uripath.join(baseURI, "STDIN") # Make abs from relative
            netStream = sys.stdin
            receivedContentType = None

    #    if diag.chatty_flag > 19: progress("HTTP Headers:" +`netStream.headers`)
    #    @@How to get at all headers??
    #    @@ Get sensible net errors and produce dignostics

        guess = None
        if receivedContentType:
            if diag.chatty_flag > 9:
                progress("Recieved Content-type: " + `receivedContentType` + " for "+addr)
            if receivedContentType.find('xml') >= 0 or (
                     receivedContentType.find('rdf')>=0
                     and not (receivedContentType.find('n3')>=0)  ):
                guess = "application/rdf+xml"
            elif receivedContentType.find('n3') >= 0:
                guess = "text/rdf+n3"
        if guess== None and contentType:
            if diag.chatty_flag > 9:
                progress("Given Content-type: " + `contentType` + " for "+addr)
            if contentType.find('xml') >= 0 or (
                    contentType.find('rdf') >= 0  and not (contentType.find('n3') >= 0 )):
                guess = "application/rdf+xml"
            elif contentType.find('n3') >= 0:
                guess = "text/rdf+n3"
            elif contentType.find('sparql') >= 0 or contentType.find('rq'):
                            guess = "x-application/sparql"
        buffer = netStream.read()
        if guess == None:

            # can't be XML if it starts with these...
            if buffer[0:1] == "#" or buffer[0:7] == "@prefix":
                guess = 'text/rdf+n3'
            elif buffer[0:6] == 'PREFIX' or buffer[0:4] == 'BASE':
                guess = "x-application/sparql"
            elif buffer.find('xmlns="') >=0 or buffer.find('xmlns:') >=0: #"
                guess = 'application/rdf+xml'
            else:
                guess = 'text/rdf+n3'
            if diag.chatty_flag > 9: progress("Guessed ContentType:" + guess)
    except (IOError, OSError):  
        raise DocumentAccessError(addr, sys.exc_info() )
        
    if asIfFrom == None:
        asIfFrom = addr
    if openFormula != None:
        F = openFormula
    else:
        F = store.newFormula()
    if topLevel:
        newTopLevelFormula(F)
    import os
    if guess == "x-application/sparql":
        if diag.chatty_flag > 49: progress("Parsing as SPARQL")
        from sparql import sparql_parser
        import sparql2cwm
        convertor = sparql2cwm.FromSparql(store, F, why=why)
        import StringIO
        p = sparql_parser.N3Parser(StringIO.StringIO(buffer), sparql_parser.branches, convertor)
        F = p.parse(sparql_parser.start).close()
    elif guess == 'application/rdf+xml':
        if diag.chatty_flag > 49: progress("Parsing as RDF")
#       import sax2rdf, xml.sax._exceptions
#       p = sax2rdf.RDFXMLParser(store, F,  thisDoc=asIfFrom, flags=flags)
        if flags == 'rdflib' or int(os.environ.get("CWM_RDFLIB", 0)):
            parser = 'rdflib'
            flags = ''
        else:
            parser = os.environ.get("CWM_RDF_PARSER", "sax2rdf")
        import rdfxml
        p = rdfxml.rdfxmlparser(store, F,  thisDoc=asIfFrom, flags=flags,
                parser=parser, why=why)

        p.feed(buffer)
        F = p.close()
    else:
        assert guess == 'text/rdf+n3'
        if diag.chatty_flag > 49: progress("Parsing as N3")
        if os.environ.get("CWM_N3_PARSER", 0) == 'n3p':
            import n3p_tm
            import triple_maker
            tm = triple_maker.TripleMaker(formula=F, store=store)
            p = n3p_tm.n3p_tm(asIfFrom, tm)
        else:
            p = notation3.SinkParser(store, F,  thisDoc=asIfFrom,flags=flags, why=why)

        try:
            p.startDoc()
            p.feed(buffer)
            p.endDoc()
        except:
            progress("Failed to parse %s" % uri or buffer)
            raise
        
    if not openFormula:
        F = F.close()
    return F 




def loadMany(store, uris, openFormula=None):
    """Get, parse and merge serveral documents, given a list of URIs. 
    
    Guesses format if necessary.
    Returns top-level formula which is the parse result.
    Raises IOError, SyntaxError
    """
    assert type(uris) is type([])
    if openFormula == None: F = store.newFormula()
    else:  F = openFormula
    f = F.uriref()
    for u in uris:
        F.reopen()  # should not be necessary
        store.load(u, openFormula=F, remember=0)
    return F.close()
    
    
    
# @@@@@@@@@@@@@ Ripped from python2.4/lib/urllib which is buggy


#  File "/devel/WWW/2000/10/swap/webAccess.py", line 104, in load
#    netStream = urlopenForRDF(addr, referer)
#  File "/devel/WWW/2000/10/swap/webAccess.py", line 72, in urlopenForRDF
#    return urllib.urlopen(addr)
#  File "/sw/lib/python2.4/urllib.py", line 77, in urlopen
#    return opener.open(url)
#  File "/sw/lib/python2.4/urllib.py", line 185, in open
#    return getattr(self, name)(url)
#  File "/sw/lib/python2.4/urllib.py", line 559, in open_data
#    f.fileno = None     # needed for addinfourl
#AttributeError: 'cStringIO.StringI' object has no attribute 'fileno'
# $ cwm 'data:text/rdf+n3;charset=utf-8;base64,QHByZWZpeCBsb2c6IDxodHRwOi8vd3d3LnczLm9yZy8yMDAwLzEwL3N3YXAvbG9nIz4gLgp7fSA9PiB7OmEgOmIgOmN9IC4g'

# Found the bug in python bug traker.
# http://sourceforge.net/tracker/index.php?func=detail&aid=1365984&group_id=5470&atid=105470
# "Fixed in revision 41548 and 41549 (2.4). by birkenfeld"
# It is in effect fixed in python 2.4.4 

def open_data(url, data=None):
    """Use "data" URL."""
    # ignore POSTed data
    #
    # syntax of data URLs:
    # dataurl   := "data:" [ mediatype ] [ ";base64" ] "," data
    # mediatype := [ type "/" subtype ] *( ";" parameter )
    # data      := *urlchar
    # parameter := attribute "=" value
    import mimetools, time
    from StringIO import StringIO
    try:
        [type, data] = url.split(',', 1)
    except ValueError:
        raise IOError, ('data error', 'bad data URL')
    if not type:
        type = 'text/plain;charset=US-ASCII'
    semi = type.rfind(';')
    if semi >= 0 and '=' not in type[semi:]:
        encoding = type[semi+1:]
        type = type[:semi]
    else:
        encoding = ''
    msg = []
    msg.append('Date: %s'%time.strftime('%a, %d %b %Y %T GMT',
                                        time.gmtime(time.time())))
    msg.append('Content-type: %s' % type)
    if encoding == 'base64':
        import base64
        data = base64.decodestring(data)
    else:
        data = unquote(data)
    msg.append('Content-length: %d' % len(data))
    msg.append('')
    msg.append(data)
    msg = '\n'.join(msg)
    f = StringIO(msg)
    headers = mimetools.Message(f, 0)
    f.fileno = None     # needed for addinfourl
    return urllib.addinfourl(f, headers, url)


    
    
    
    
#@@@@@@@@@@  Junk - just to keep track iof the interface to sandros stuff and rdflib
    
def getParser(format, inputURI, workingContext, flags):
    """Return something which can load from a URI in the given format, while
    writing to the given store.
    """
    r = BecauseOfCommandLine(sys.argv[0]) # @@ add user, host, pid, date time? Privacy!
    if format == "rdf" :
        touch(_store)
        if "l" in flags["rdf"]:
            from rdflib2rdf import RDFXMLParser
        else:
            rdfParserName = os.environ.get("CWM_RDF_PARSER", "sax2rdf")
            if rdfParserName == "rdflib2rdf":
                from rdflib2rdf import RDFXMLParser
            elif rdfParserName == "sax2rdf":
                from sax2rdf import RDFXMLParser
            else:
                raise RuntimeError("Unknown RDF parser: " + rdfParserName)
        return RDFXMLParser(_store, workingContext, inputURI,
                                        flags=flags[format], why=r)
    elif format == "n3":
        touch(_store)
        return notation3.SinkParser(_store, openFormula=workingContext,
                    thisDoc=inputURI,  why=r)
    else:
        need(lxkb)
        touch(lxkb)
        return LX.language.getParser(language=format,
                                     sink=lxkb,
                                     flags=flags)

class DocumentAccessError(IOError):
    def __init__(self, uri, info):
        self._uri = uri
        self._info = info
        
    def __str__(self):
        # See C:\Python16\Doc\ref\try.html or URI to that effect
#        reason = `self._info[0]` + " with args: " + `self._info[1]`
        reason = indentString(self._info[1].__str__())
        return ("Unable to access document <%s>, because:\n%s" % ( self._uri, reason))
    

