#!/bin/env python
"""
Uniform Resource Identifier (URI) path manipulation,
above the access layer

The name of this module and the functions are somewhat
arbitrary; they hark to other parts of the python
library; e.g. uripath.join() is somewhat like os.path.join().

REFERENCES

  Uniform Resource Identifiers (URI): Generic Syntax
  http://www.ietf.org/rfc/rfc2396.txt

  The Web Model: Information hiding and URI syntax (Jan 98)
  http://www.w3.org/DesignIssues/Model.html

  URI API design [was: URI Test Suite] Dan Connolly (Sun, Aug 12 2001)
  http://lists.w3.org/Archives/Public/uri/2001Aug/0021.html

"""

__version__ = "$Id: uripath.py,v 1.21 2007/06/26 02:36:16 syosi Exp $"

from string import find, rfind, index


def splitFrag(uriref):
    """split a URI reference between the fragment and the rest.

    Punctuation is thrown away.

    e.g.
    
    >>> splitFrag("abc#def")
    ('abc', 'def')

    >>> splitFrag("abcdef")
    ('abcdef', None)

    """

    i = rfind(uriref, "#")
    if i>= 0: return uriref[:i], uriref[i+1:]
    else: return uriref, None

def splitFragP(uriref, punct=0):
    """split a URI reference before the fragment

    Punctuation is kept.
    
    e.g.

    >>> splitFragP("abc#def")
    ('abc', '#def')

    >>> splitFragP("abcdef")
    ('abcdef', '')

    """

    i = rfind(uriref, "#")
    if i>= 0: return uriref[:i], uriref[i:]
    else: return uriref, ''


def join(here, there):
    """join an absolute URI and URI reference
    (non-ascii characters are supported/doctested;
    haven't checked the details of the IRI spec though)

    here is assumed to be absolute.
    there is URI reference.

    >>> join('http://example/x/y/z', '../abc')
    'http://example/x/abc'

    Raise ValueError if there uses relative path
    syntax but here has no hierarchical path.

    >>> join('mid:foo@example', '../foo')
    Traceback (most recent call last):
        raise ValueError, here
    ValueError: Base <mid:foo@example> has no slash after colon - with relative '../foo'.

    >>> join('http://example/x/y/z', '')
    'http://example/x/y/z'
    
    >>> join('mid:foo@example', '#foo')
    'mid:foo@example#foo'
    
    We grok IRIs

    >>> len(u'Andr\\xe9')
    5
    
    >>> join('http://example.org/', u'#Andr\\xe9')
    u'http://example.org/#Andr\\xe9'
    """

    assert(find(here, "#") < 0), "Base may not contain hash: '%s'"% here # caller must splitFrag (why?)

    slashl = find(there, '/')
    colonl = find(there, ':')

    # join(base, 'foo:/') -- absolute
    if colonl >= 0 and (slashl < 0 or colonl < slashl):
        return there

    bcolonl = find(here, ':')
    assert(bcolonl >= 0), "Base uri '%s' is not absolute" % here # else it's not absolute

    path, frag = splitFragP(there)
    if not path: return here + frag
    
    # join('mid:foo@example', '../foo') bzzt
    if here[bcolonl+1:bcolonl+2] <> '/':
        raise ValueError ("Base <%s> has no slash after colon - with relative '%s'." %(here, there))

    if here[bcolonl+1:bcolonl+3] == '//':
        bpath = find(here, '/', bcolonl+3)
    else:
        bpath = bcolonl+1

    # join('http://xyz', 'foo')
    if bpath < 0:
        bpath = len(here)
        here = here + '/'

    # join('http://xyz/', '//abc') => 'http://abc'
    if there[:2] == '//':
        return here[:bcolonl+1] + there

    # join('http://xyz/', '/abc') => 'http://xyz/abc'
    if there[:1] == '/':
        return here[:bpath] + there

    slashr = rfind(here, '/')

    while 1:
        if path[:2] == './':
            path = path[2:]
        if path == '.':
            path = ''
        elif path[:3] == '../' or path == '..':
            path = path[3:]
            i = rfind(here, '/', bpath, slashr)
            if i >= 0:
                here = here[:i+1]
                slashr = i
        else:
            break

    return here[:slashr+1] + path + frag


    
import re
import string
commonHost = re.compile(r'^[-_a-zA-Z0-9.]+:(//[^/]*)?/[^/]*$')


def refTo(base, uri):
    """figure out a relative URI reference from base to uri

    >>> refTo('http://example/x/y/z', 'http://example/x/abc')
    '../abc'

    >>> refTo('file:/ex/x/y', 'file:/ex/x/q/r#s')
    'q/r#s'
    
    >>> refTo(None, 'http://ex/x/y')
    'http://ex/x/y'

    >>> refTo('http://ex/x/y', 'http://ex/x/y')
    ''

    Note the relationship between refTo and join:
    join(x, refTo(x, y)) == y
    which points out certain strings which cannot be URIs. e.g.
    >>> x='http://ex/x/y';y='http://ex/x/q:r';join(x, refTo(x, y)) == y
    0

    So 'http://ex/x/q:r' is not a URI. Use 'http://ex/x/q%3ar' instead:
    >>> x='http://ex/x/y';y='http://ex/x/q%3ar';join(x, refTo(x, y)) == y
    1
    
    This one checks that it uses a root-realtive one where that is
    all they share.  Now uses root-relative where no path is shared.
    This is a matter of taste but tends to give more resilience IMHO
    -- and shorter paths

    Note that base may be None, meaning no base.  In some situations, there
    just ain't a base. Slife. In these cases, relTo returns the absolute value.
    The axiom abs(,rel(b,x))=x still holds.
    This saves people having to set the base to "bogus:".

    >>> refTo('http://ex/x/y/z', 'http://ex/r')
    '/r'

    """

#    assert base # don't mask bugs -danc # not a bug. -tim
    if not base: return uri
    if base == uri: return ""
    
    # Find how many path segments in common
    i=0
    while i<len(uri) and i<len(base):
        if uri[i] == base[i]: i = i + 1
        else: break
    # print "# relative", base, uri, "   same up to ", i
    # i point to end of shortest one or first difference

    m = commonHost.match(base[:i])
    if m:
        k=uri.find("//")
        if k<0: k=-2 # no host
        l=uri.find("/", k+2)
        if uri[l+1:l+2] != "/" and base[l+1:l+2] != "/" and uri[:l]==base[:l]:
            return uri[l:]

    if uri[i:i+1] =="#" and len(base) == i: return uri[i:] # fragment of base

    while i>0 and uri[i-1] != '/' : i=i-1  # scan for slash

    if i < 3: return uri  # No way.
    if string.find(base, "//", i-2)>0 \
       or string.find(uri, "//", i-2)>0: return uri # An unshared "//"
    if string.find(base, ":", i)>0: return uri  # An unshared ":"
    n = string.count(base, "/", i)
    if n == 0 and i<len(uri) and uri[i] == '#':
        return "./" + uri[i:]
    elif n == 0 and i == len(uri):
        return "./"
    else:
        return ("../" * n) + uri[i:]

import os
def base():
        """The base URI for this process - the Web equiv of cwd
        
        Relative or abolute unix-standard filenames parsed relative to
        this yeild the URI of the file.
        If we had a reliable way of getting a computer name,
        we should put it in the hostname just to prevent ambiguity

        """
#       return "file://" + hostname + os.getcwd() + "/"
        return "file://" + _fixslash(os.getcwd()) + "/"


def _fixslash(str):
    """ Fix windowslike filename to unixlike - (#ifdef WINDOWS)"""
    s = str
    for i in range(len(s)):
        if s[i] == "\\": s = s[:i] + "/" + s[i+1:]
    if s[0] != "/" and s[1] == ":": s = s[2:]  # @@@ Hack when drive letter present
    return s

URI_unreserved = "ABCDEFGHIJJLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
    # unreserved  = ALPHA / DIGIT / "-" / "." / "_" / "~"
    
def canonical(str_in):
    """Convert equivalent URIs (or parts) to the same string
    
    There are many differenet levels of URI canonicalization
    which are possible.  See http://www.ietf.org/rfc/rfc3986.txt
    Done:
    - Converfting unicode IRI to utf-8
    - Escaping all non-ASCII
    - De-escaping, if escaped, ALPHA (%41-%5A and %61-%7A), DIGIT (%30-%39),
      hyphen (%2D), period (%2E), underscore (%5F), or tilde (%7E) (Sect 2.4) 
    - Making all escapes uppercase hexadecimal
    Not done:
    - Making URI scheme lowercase
    - changing /./ or  /foo/../ to / with care not to change host part
    
    
    >>> canonical("foo bar")
    'foo%20bar'
    
    >>> canonical(u'http:')
    'http:'
    
    >>> canonical('fran%c3%83%c2%a7ois')
    'fran%C3%83%C2%A7ois'
    
    >>> canonical('a')
    'a'
    
    >>> canonical('%4e')
    'N'

    >>> canonical('%9d')
    '%9D'
    
    >>> canonical('%2f')
    '%2F'

    >>> canonical('%2F')
    '%2F'

    """
    if type(str_in) == type(u''):
        s8 = str_in.encode('utf-8')
    else:
        s8 = str_in
    s = ''
    i = 0
    while i < len(s8):
        ch = s8[i]; n = ord(ch)
        if (n > 126) or (n < 33) :   # %-encode controls, SP, DEL, and utf-8
            s += "%%%02X" % ord(ch)
        elif ch == '%' and i+2 < len(s8):
            ch2 = s8[i+1:i+3].decode('hex')
            if ch2 in URI_unreserved: s += ch2
            else: s += "%%%02X" % ord(ch2)
            i = i+3
            continue
        else:
            s += ch
        i = i +1
    return s
    
    
import unittest

class Tests(unittest.TestCase):
    def testPaths(self):
        cases = (("foo:xyz", "bar:abc", "bar:abc"),
                 ('http://example/x/y/z', 'http://example/x/abc', '../abc'),
                 ('http://example2/x/y/z', 'http://example/x/abc', 'http://example/x/abc'),
                 ('http://ex/x/y/z', 'http://ex/x/r', '../r'),
                 #             ('http://ex/x/y/z', 'http://ex/r', '../../r'),    # DanC had this.
                 ('http://ex/x/y', 'http://ex/x/q/r', 'q/r'),
                 ('http://ex/x/y', 'http://ex/x/q/r#s', 'q/r#s'),
                 ('http://ex/x/y', 'http://ex/x/q/r#s/t', 'q/r#s/t'),
                 ('http://ex/x/y', 'ftp://ex/x/q/r', 'ftp://ex/x/q/r'),
                 ('http://ex/x/y', 'http://ex/x/y', ''),
                 ('http://ex/x/y/', 'http://ex/x/y/', ''),
                 ('http://ex/x/y/pdq', 'http://ex/x/y/pdq', ''),
                 ('http://ex/x/y/', 'http://ex/x/y/z/', 'z/'),
                 ('file:/swap/test/animal.rdf', 'file:/swap/test/animal.rdf#Animal', '#Animal'),
                 ('file:/e/x/y/z', 'file:/e/x/abc', '../abc'),
                 ('file:/example2/x/y/z', 'file:/example/x/abc', '/example/x/abc'),   # TBL
                 ('file:/ex/x/y/z', 'file:/ex/x/r', '../r'),
                 ('file:/ex/x/y/z', 'file:/r', '/r'),        # I prefer this. - tbl
                 ('file:/ex/x/y', 'file:/ex/x/q/r', 'q/r'),
                 ('file:/ex/x/y', 'file:/ex/x/q/r#s', 'q/r#s'),
                 ('file:/ex/x/y', 'file:/ex/x/q/r#', 'q/r#'),
                 ('file:/ex/x/y', 'file:/ex/x/q/r#s/t', 'q/r#s/t'),
                 ('file:/ex/x/y', 'ftp://ex/x/q/r', 'ftp://ex/x/q/r'),
                 ('file:/ex/x/y', 'file:/ex/x/y', ''),
                 ('file:/ex/x/y/', 'file:/ex/x/y/', ''),
                 ('file:/ex/x/y/pdq', 'file:/ex/x/y/pdq', ''),
                 ('file:/ex/x/y/', 'file:/ex/x/y/z/', 'z/'),
                 ('file:/devel/WWW/2000/10/swap/test/reluri-1.n3', 
                  'file://meetings.example.com/cal#m1', 'file://meetings.example.com/cal#m1'),
                 ('file:/home/connolly/w3ccvs/WWW/2000/10/swap/test/reluri-1.n3', 'file://meetings.example.com/cal#m1', 'file://meetings.example.com/cal#m1'),
                 ('file:/some/dir/foo', 'file:/some/dir/#blort', './#blort'),
                 ('file:/some/dir/foo', 'file:/some/dir/#', './#'),

                 # From Graham Klyne Thu, 20 Feb 2003 18:08:17 +0000
                 ("http://example/x/y%2Fz", "http://example/x/abc", "abc"),
                 ("http://example/x/y/z", "http://example/x%2Fabc", "/x%2Fabc"),
                 ("http://example/x/y%2Fz", "http://example/x%2Fabc", "/x%2Fabc"),
                 ("http://example/x%2Fy/z", "http://example/x%2Fy/abc", "abc"),
                 # Ryan Lee
                 ("http://example/x/abc.efg", "http://example/x/", "./")
                 )

        for inp1, inp2, exp in cases:
            self.assertEquals(refTo(inp1, inp2), exp)
            self.assertEquals(join(inp1, exp), inp2)


    def testSplit(self):
        cases = (
            ("abc#def", "abc", "def"),
            ("abc", "abc", None),
            ("#def", "", "def"),
            ("", "", None),
            ("abc#de:f", "abc", "de:f"),
            ("abc#de?f", "abc", "de?f"),
            ("abc#de/f", "abc", "de/f"),
            )
        for inp, exp1, exp2 in cases:
            self.assertEquals(splitFrag(inp), (exp1, exp2))

    def testRFCCases(self):

        base = 'http://a/b/c/d;p?q'

        # C.1.  Normal Examples

        normalExamples = (
            (base, 'g:h', 'g:h'),
            (base, 'g', 'http://a/b/c/g'),
            (base, './g', 'http://a/b/c/g'),
            (base, 'g/', 'http://a/b/c/g/'),
            (base, '/g', 'http://a/g'),
            (base, '//g', 'http://g'),
            (base, '?y', 'http://a/b/c/?y'), #@@wow... really?
            (base, 'g?y', 'http://a/b/c/g?y'),
            (base, '#s', 'http://a/b/c/d;p?q#s'), #@@ was: (current document)#s
            (base, 'g#s', 'http://a/b/c/g#s'),
            (base, 'g?y#s', 'http://a/b/c/g?y#s'),
            (base, ';x', 'http://a/b/c/;x'),
            (base, 'g;x', 'http://a/b/c/g;x'),
            (base, 'g;x?y#s', 'http://a/b/c/g;x?y#s'),
            (base, '.', 'http://a/b/c/'),
            (base, './', 'http://a/b/c/'),
            (base, '..', 'http://a/b/'),
            (base, '../', 'http://a/b/'),
            (base, '../g', 'http://a/b/g'),
            (base, '../..', 'http://a/'),
            (base, '../../', 'http://a/'),
            (base, '../../g', 'http://a/g')
            )
        
        otherExamples = (
            (base, '', base),
            (base, '../../../g', 'http://a/g'), #@@disagree with RFC2396
            (base, '../../../../g', 'http://a/g'), #@@disagree with RFC2396
            (base, '/./g', 'http://a/./g'),
            (base, '/../g', 'http://a/../g'),
            (base, 'g.', 'http://a/b/c/g.'),
            (base, '.g', 'http://a/b/c/.g'),
            (base, 'g..', 'http://a/b/c/g..'),
            (base, '..g', 'http://a/b/c/..g'),
            
            (base, './../g', 'http://a/b/g'),
            (base, './g/.', 'http://a/b/c/g/.'), #@@hmmm...
            (base, 'g/./h', 'http://a/b/c/g/./h'), #@@hmm...
            (base, 'g/../h', 'http://a/b/c/g/../h'),
            (base, 'g;x=1/./y', 'http://a/b/c/g;x=1/./y'), #@@hmmm...
            (base, 'g;x=1/../y', 'http://a/b/c/g;x=1/../y'),  #@@hmmm...
            
            (base, 'g?y/./x', 'http://a/b/c/g?y/./x'),
            (base, 'g?y/../x', 'http://a/b/c/g?y/../x'),
            (base, 'g#s/./x', 'http://a/b/c/g#s/./x'),
            (base, 'g#s/../x', 'http://a/b/c/g#s/../x')
            )
        
        for b, inp, exp in normalExamples + otherExamples:
            if exp is None:
                self.assertRaises(ValueError, join, b, inp)
            else:
                self.assertEquals(join(b, inp), exp)

def _test():
    import doctest, uripath
    doctest.testmod(uripath)
    unittest.main()

if __name__ == '__main__':
    _test()


# $Log: uripath.py,v $
# Revision 1.21  2007/06/26 02:36:16  syosi
# fix tabs
#
# Revision 1.20  2007/01/25 20:26:50  timbl
# @@@ BEWARE UNTESTED PARTIAL VERSION -- Introducing XML Literals as DOM objects
#
# Revision 1.19  2006/11/09 22:44:12  connolly
# start base with file:// rather than just file:/
# for interop with xsltproc
#
# Revision 1.18  2006/07/07 22:06:50  connolly
# fix bug with joining mid:abc with #foo
#
# Revision 1.17  2006/06/17 19:27:27  timbl
# Add canonical() with unit tests
#
# Revision 1.16  2004/03/21 04:24:35  timbl
# (See doc/changes.html)
# on xml output, nodeID was incorrectly spelled.
# update.py provides cwm's --patch option.
# diff.py as independent progrem generates patch files for cwm --patch
#
# Revision 1.15  2004/01/28 22:22:10  connolly
# tested that IRIs work in uripath.join()
#
# Revision 1.14  2003/10/20 17:31:55  timbl
# Added @keyword support.
# (eventually got python+expat to wrok on fink, with patch)
# Trig functions are in, thanks to Karl, with some changes, but NOT in regeression.n3
# see test/math/test-trigo.n3 for now.
#
# Revision 1.13  2003/07/03 21:04:39  timbl
# New string function to compare strings normalizing case and whitespace string:containsRoughly
#
# Revision 1.12  2003/04/03 22:35:12  ryanlee
# fixed previous fix, added test case
#
# Revision 1.11  2003/04/03 22:06:54  ryanlee
# small fix in if, line 217
#
# Revision 1.10  2003/02/24 15:06:38  connolly
# some more tests from Graham
#
# Revision 1.9  2002/12/25 20:01:32  timbl
# some --flatten tests fail. --why fails. Formulae must be closed to be referenced in a add()
#
# Revision 1.8  2002/11/24 03:12:02  timbl
# base can be None in uripath:refTo
#
# Revision 1.7  2002/09/04 05:03:07  connolly
# convertet unittests to use python doctest and unittest modules; cleaned up docstrings a bit
#
# Revision 1.6  2002/09/04 04:07:50  connolly
# fixed uripath.refTo
#
# Revision 1.5  2002/08/23 04:36:15  connolly
# fixed refTo case: file:/some/dir/foo  ->  file:/some/dir/#blort
#
# Revision 1.4  2002/08/07 14:32:21  timbl
# uripath changes. passes 51 general tests and 25 loopback tests
#
# Revision 1.3  2002/08/06 01:36:09  connolly
# cleanup: diagnostic interface, relative/absolute uri handling
#
# Revision 1.2  2002/03/15 23:53:02  connolly
# handle no-auth case
#
# Revision 1.1  2002/02/19 22:52:42  connolly
# renamed uritools.py to uripath.py
#
# Revision 1.2  2002/02/18 07:33:51  connolly
# pathTo seems to work
#
