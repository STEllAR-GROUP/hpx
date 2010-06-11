#!/usr/bin/python 
"""
Matematical Built-Ins for CWM/Llyn using Strings

Allows CWM to do addition, multiplication, subtraction, division, 
remainders, negation, exponentiation, count the members in a DAML 
list, and do the normal truth checking functions, only sub classed 
for numeric values.

All values are represented by string types (untyped literals in RDF).
See math (no s) for versions using numeric types.

cf. http://www.w3.org/2000/10/swap/cwm.py and 
http://ilrt.org/discovery/chatlogs/rdfig/2001-12-01.txt from 
"01:20:58" onwards.
"""

__author__ = 'Sean B. Palmer'
__cvsid__ = '$Id: cwm_maths.py,v 1.11 2005/06/09 21:05:14 syosi Exp $'
__version__ = '$Revision: 1.11 $'

import sys, string, re, urllib

from term import LightBuiltIn, Function, ReverseFunction
from local_decimal import Decimal

MATHS_NS_URI = 'http://www.w3.org/2000/10/swap/maths#'

def tidy(x):
    #DWC bugfix: "39.03555" got changed to "393555"
    if x == None: return None
    s = str(x)
    if s[-2:] == '.0': s=s[:-2]
    return s


def isString(x):
    # in 2.2, evidently we can test for isinstance(types.StringTypes)
    return type(x) is type('') or type(x) is type(u'')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# M A T H E M A T I C A L   B U I L T - I N s
#
# Some mathematical built-ins: the heaviest one gets the amount of list 
# members in a DAML list.
# 
# Thanks to deltab, bijan, and oierw for helping me to name the 
# properties, and to TimBL for CWM and the built-in templates in the 
# first place.
#
# Light Built-in classes - these are all reverse functions

# add, take, multiply, divide


class BI_absoluteValue(LightBuiltIn, Function):
    def evaluateObject(self, x):
            t = abs(Decimal(x))
            if t is not None: return tidy(t)

class BI_rounded(LightBuiltIn, Function):
    def evaluateObject(self, x):
            t = round(float(x))
            if t is not None: return tidy(t)

class BI_sum(LightBuiltIn, Function):
    def evaluateObject(self,  subj_py): 
        t = 0
        for x in subj_py:
            if not isString(x): return None
            t += Decimal(x)
        return tidy(t)

class BI_sumOf(LightBuiltIn, ReverseFunction):
    def evaluateSubject(self, obj_py): 
        t = 0
        for x in obj_py: t += Decimal(x)
        return tidy(t)


class BI_difference(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        t = None
        if len(subj_py) == 2: t = Decimal(subj_py[0]) - Decimal(subj_py[1])
        return tidy(t)

class BI_differenceOf(LightBuiltIn, ReverseFunction):
    def evaluateSubject(self,  obj_py): 
        t = None
        if len(obj_py) == 2: t = Decimal(obj_py[0]) - Decimal(obj_py[1])
        return tidy(t)

class BI_product(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        t = 1
        for x in subj_py: t *= Decimal(x)
        return tidy(t)

class BI_factors(LightBuiltIn, ReverseFunction):
    def evaluateSubject(self, obj_py): 
        t = 1
        for x in obj_py: t *= Decimal(x)
        return tidy(t)

class BI_quotient(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        t = None
        if len(subj_py) == 2: t = float(subj_py[0]) / float(subj_py[1])
        return tidy(t)

class BI_integerQuotient(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        t = None
        if len(subj_py) == 2: t = long(subj_py[0]) / long(subj_py[1])
        return tidy(t)

class BI_quotientOf(LightBuiltIn, ReverseFunction):
    def evaluateSubject(self,  obj_py): 
        t = None
        if len(obj_py) == 2: t = float(obj_py[0]) / float(obj_py[1])
        return tidy(t)

# remainderOf and negationOf

class BI_remainder(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        t = None
        if len(subj_py) == 2: t = float(subj_py[0]) % float(subj_py[1])
        return tidy(t)

class BI_remainderOf(LightBuiltIn, ReverseFunction):
    def evaluateSubject(self,  obj_py): 
        t = None
        if len(obj_py) == 2: t = float(obj_py[0]) % float(obj_py[1])
        return tidy(t)

class BI_negation(LightBuiltIn, Function, ReverseFunction):

    def evalaluateObject(self, subject):
            t = -Decimal(subject)
            if t is not None: return tidy(t)

    def evalaluateSubject(self, object):
            t = -Decimal(object)
            if t is not None: return tidy(t)

# Power

class BI_exponentiation(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        t = None
        if len(subj_py) == 2: t = float(subj_py[0]) ** float(subj_py[1])
        return tidy(t)

class BI_exponentiationOf(LightBuiltIn, ReverseFunction):

    def evaluateSubject(self, obj_py): 
        t = None
        if len(obj_py) == 2: t = float(obj_py[0]) ** float(obj_py[1])
        return tidy(t)

# Math greater than and less than etc., modified from cwm_string.py
# These are truth testing things  - Binary logical operators

class BI_greaterThan(LightBuiltIn):
    def eval(self, subj, obj, queue, bindings, proof, query):
        return (float(subj.string) > float(obj.string))

class BI_notGreaterThan(LightBuiltIn):
    def eval(self, subj, obj, queue, bindings, proof, query):
        return (float(subj.string) <= float(obj.string))

class BI_lessThan(LightBuiltIn):
    def eval(self, subj, obj, queue, bindings, proof, query):
        return (float(subj.string) < float(obj.string))

class BI_notLessThan(LightBuiltIn):
    def eval(self, subj, obj, queue, bindings, proof, query):
        return (float(subj.string) >= float(obj.string))

class BI_equalTo(LightBuiltIn):
    def eval(self, subj, obj, queue, bindings, proof, query):
        return (float(subj.string) == float(obj.string))

class BI_notEqualTo(LightBuiltIn):
    def eval(self, subj, obj, queue, bindings, proof, query):
        return (float(subj.string) != float(obj.string))

# memberCount - this is a proper forward function

class BI_memberCount(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        t = len(subj_py)
        return tidy(t)

#  Register the string built-ins with the store

def register(store):
    str = store.symbol(MATHS_NS_URI[:-1])
    str.internFrag('sum', BI_sum)
    str.internFrag('difference', BI_difference)
    str.internFrag('product', BI_product)
    str.internFrag('quotient', BI_quotient)
    str.internFrag('integerQuotient', BI_integerQuotient)
    str.internFrag('remainder', BI_remainder)
    str.internFrag('exponentiation', BI_exponentiation)

    str.internFrag('sumOf', BI_sumOf)
    str.internFrag('differenceOf', BI_differenceOf)
    str.internFrag('factors', BI_factors)
    str.internFrag('quotientOf', BI_quotientOf)
    str.internFrag('remainderOf', BI_remainderOf)
    str.internFrag('exponentiationOf', BI_exponentiationOf)

    str.internFrag('negation', BI_negation)
    str.internFrag('absoluteValue', BI_absoluteValue)
    str.internFrag('rounded', BI_rounded)

    str.internFrag('greaterThan', BI_greaterThan)
    str.internFrag('notGreaterThan', BI_notGreaterThan)
    str.internFrag('lessThan', BI_lessThan)
    str.internFrag('notLessThan', BI_notLessThan)
    str.internFrag('equalTo', BI_equalTo)
    str.internFrag('notEqualTo', BI_notEqualTo)
    str.internFrag('memberCount', BI_memberCount)

if __name__=="__main__": 
   print string.strip(__doc__)
