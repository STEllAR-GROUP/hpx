#! /usr/bin/python
"""


$Id: cwm_os.py,v 1.12 2007/06/26 02:36:15 syosi Exp $

Operating systems built-ins for cwm
http://www.w3.org/2000/10/swap/string.py

See cwm.py and the os module in python

"""

import os

from term import LightBuiltIn, Function, ReverseFunction
from diag import verbosity, progress
import uripath


OS_NS_URI = "http://www.w3.org/2000/10/swap/os#"



###############################################################################################
#
#                              O P E R A T I N G   S Y T E M   B U I L T - I N s
#
#
#   Light Built-in classes

# Read Operating sytem environment lookup - read-only
#
# Not fatal if not defined
class BI_environ(LightBuiltIn, Function):
    def evaluateObject(self,  subj_py):
        if isString(subj_py): return os.environ.get(subj_py, None)
        progress("os:environ input is not a string: "+`subj_py`)

class BI_baseAbsolute(LightBuiltIn, Function):
    """The baseAbsolute function generates an absolute URIref from a string,
    interpreting the string as a a relative URIref relative to the current
    process base URI (typically, current working directory).
    It is not a reverse function, because sereral different relativisations
    exist for the same absolute URI. See uripath.py."""
    def evaluateObject(self, subj_py):
        if verbosity() > 80: progress("os:baseAbsolute input:"+`subj_py`)
        if isString(subj_py):
            return uripath.join(uripath.base(), subj_py)
        progress("Warning: os:baseAbsolute input is not a string: "+`subj_py`)

class BI_baseRelative(LightBuiltIn, Function, ReverseFunction):
    """The baseRelative of a URI is its expression relation to the process base URI.
    It is 1:1, being an arbitrary cannonical form.
    It is a reverse function too, as you can always work the other way."""
    def evaluateObject(self, subj_py):
        if verbosity() > 80: progress("os:baseRelative input:"+`subj_py`)
        if isString(subj_py):
            return uripath.refTo(uripath.base(), subj_py)
        progress("Warning: os:baseRelative input is not a string: "+`subj_py`)

    def evaluateSubject(self, subj_py):
        return BI_baseAbsolute.evaluateObject(self, subj_py)

# Command line argument: read-only
#  The command lines are passed though cwm using "--with" and into the RDFStore when init'ed.
# Not fatal if not defined
class BI_argv(LightBuiltIn, Function):
    def evaluateObject(self,  subj_py):
        if verbosity() > 80: progress("os:argv input:"+`subj_py`)
        if  self.store.argv:  # Not None or []. was also: isString(subj_py) and
            try:
                argnum = int(subj_py) -1
            except ValueError:
                if verbosity() > 30:
                    progress("os:argv input is not a number: "+`subj_py`)
                return None
            if argnum < len(self.store.argv):
                return self.store.argv[argnum]

def isString(x):
    # in 2.2, evidently we can test for isinstance(types.StringTypes)
    return type(x) is type('') or type(x) is type(u'')

#  Register the string built-ins with the store

def register(store):
    str = store.symbol(OS_NS_URI[:-1])
    str.internFrag("environ", BI_environ)
    str.internFrag("baseRelative", BI_baseRelative)
    str.internFrag("baseAbsolute", BI_baseAbsolute)
    str.internFrag("argv", BI_argv)

# ends

