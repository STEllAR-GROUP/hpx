#! /usr/bin/python
"""
$Id: myStore.py,v 1.14 2007/08/06 16:13:56 syosi Exp $

Process-global store

Global variables are often a bad idea. However, the majority of cwm
applications involve just one RDF store. One store can contain many
formulae.  The main thing which these formulae of the same store share
is the index with which names and strings are interned.  Within a
store, you can compare things by comparing memory addresses rather
than the whole tring, uri or list.

Therefore, it is normal to just use one store.  When you do this, the
store paremeter to most methods beceomes unnecessary, and you get a
simpler API.  That is what this module does. If you import it, you
will get a global store. This will not stop you using other stores.

You also get the Namespace() class which allows you to generate
symbols easily.

History:
    Spilt off from  thing.py 2003-08-19

$Log: myStore.py,v $
Revision 1.14  2007/08/06 16:13:56  syosi
A month of changes

Revision 1.13  2007/06/26 02:36:15  syosi
fix tabs

Revision 1.12  2005/10/24 16:58:38  timbl
--n3=B flag introduced. --why improved but not perfect.

Revision 1.11  2004/08/08 01:44:49  syosi
undo what I did last thing Friday. Those changes are nowhere near ready for the trunk

Revision 1.9  2004/07/29 16:16:11  syosi
rdflib tests in the default test suite

Revision 1.8  2004/04/19 13:32:22  connolly
trap __special_names__ in Namespace.__getattr__ so
that pychecker can work

Revision 1.7  2004/03/21 04:24:35  timbl
(See doc/changes.html)
on xml output, nodeID was incorrectly spelled.
update.py provides cwm's --patch option.
diff.py as independent progrem generates patch files for cwm --patch

Revision 1.6  2004/03/09 23:55:50  connolly
updated load to track change in llyn

Revision 1.5  2004/03/06 20:39:38  timbl
See http://www.w3.org/2000/10/swap/doc/changes.html for details
- Regresssion test incorporates the RDF Core Positive Parser Tests except XMLLiteral & reification
- xml:base support was added in the parser.
- Use the --rdf=R flag to allow RDF to be parsed even when there is no enveloping <rdf:RDF> tag
- nodeid generated on RDF output
- Automatically generated terms with no URIs sort after anything which has a URI.
- Namespace prefix smarts on output - default ns used for that most frequently used.
- suppresses namespace prefix declarations which are not actually needed in the output.
- Cwm will also make up prefixes when it needs them for a namespace, and none of the input data uses one.-
- Will not use namespace names for URIs which do not have a "#". Including a "/" in the flags overrides.

Revision 1.4  2004/01/29 21:10:39  timbl
ooops - ref to SYMBOL

Revision 1.3  2004/01/28 23:03:00  connolly
- added unit tests to confirm that symbol functions take ustrings
- wrapped some comments at 79 chars
  per http://www.python.org/doc/essays/styleguide.html


"""

import uripath

# Allow a strore provdier to register:

store = None
storeClass = None

def setStoreClass(c):
    """Set the process-global class to be used to generate a new store if needed"""
    global storeClass
    storeClass = c

def setStore(s):
    """Set the process-global default store to be used when an explicit store is not"""
    global store
    store = s

def _checkStore(s=None):
    """Check that an explict or implicit stroe exists"""
    global store, storeClass
    if s != None: return s
    if store != None: return store
    if storeClass == None:
        import llyn   # default 
    assert storeClass!= None, "Some storage module must register with myStore.py before you can use it"
    store = storeClass() # Make new one
    return store


def symbol(uri):
    """Create or reuse an interned version of the given symbol
    in the default store. and return it for future use

    >>> x = symbol(u'http://example.org/#Andr\\xe9')
    >>> y = symbol(u'http://example.org/#Andr\\xe9')
    >>> x is y
    1
    """
    return _checkStore().newSymbol(uri)
    
def literal(str, dt=None, lang=None):
    """Create or reuse, in the default store, an interned version of
    the given literal string and return it for future use

    >>> x = literal("#Andr\\xe9")
    >>> y = literal("#Andr\\xe9")
    >>> x is y
    1

    """
    
    return _checkStore().newLiteral(str, dt, lang)


def intern(v):
    return _checkStore().intern(v)

def formula():
    """Create or reuse, in the default store, a new empty formula (triple people think: triple store)
    and return it for future use"""
    return _checkStore().newFormula()

#def bNode(str, context):
#    """Create or reuse, in the default store, a new unnamed node within the given
#    formula as context, and return it for future use"""
#    return _checkStore().newBlankNode(context)

def existential(str, context, uri):
    """Create or reuse, in the default store, a new named variable
    existentially qualified within the given
    formula as context, and return it for future use"""
    return _checkStore().newExistential(context, uri)

def universal(str, context, uri):
    """Create or reuse, in the default store, a named variable
    universally qualified within the given
    formula as context, and return it for future use"""
    return _checkStore().newUniversal(context, uri)

def load(uri=None, openFormula=None, contentType=None, remember=1, flags=""):
    """Get and parse document.  Guesses format if necessary.

    uri:      if None, load from standard input.
    remember: if 1, store as metadata the relationship between this URI and this formula.
    
    Returns:  top-level formula of the parsed document.
    Raises:   IOError, SyntaxError, DocumentError
    """
    return _checkStore().load(uri, openFormula=openFormula, contentType=contentType,
                        remember=remember, flags=flags)

def loadMany(uris, openFormula=None, referer=None):
    """Load a number of resources into the same formula
    
    Returns:  top-level formula of the parsed information.
    Raises:   IOError, SyntaxError, DocumentError
    """
    return _checkStore().loadMany(uris, openFormula, referer=referer)

def bind(prefix, uri):
    return _checkStore().bind(prefix, uri)

class Namespace(object):
    """A shortcut for getting a symbols as interned by the default store

      >>> RDF = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
      >>> x = RDF.type
      >>> y = RDF.type
      >>> x is y
      1

    """
    
    def __init__(self, name, store=None):
        if ':' not in name:    #, "must be absolute: %s" % name
            base = uripath.base()
            name = uripath.join(base, name)
        self._name = name
        self.store = store
        self._seen = {}
    
    def __getattr__(self, lname):
        """get the lname Symbol in this namespace.

        lname -- an XML name (limited to URI characters)
        I hope this is only called *after* the ones defines above have been checked
        """
        if lname.startswith("__"): # python internal
            raise AttributeError, lname
        
        return _checkStore(self.store).symbol(self._name+lname)

    def sym(self, lname):
        """For getting a symbol for an expression, rather than a constant.
        For, and from, pim/toIcal.py"""
        return  _checkStore(self.store).symbol(self._name + lname)
    __getitem__ = sym



def _test():
    import llyn
    store = llyn.RDFStore()
    setStore(store)
    
    import doctest, myStore
    return doctest.testmod(myStore)
     
if __name__ == "__main__":
    _test()
