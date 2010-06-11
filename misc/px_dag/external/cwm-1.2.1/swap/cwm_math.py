#!/usr/bin/python 
"""
Matematical Built-Ins for CWM/Llyn

Allows CWM to do addition, multiplication, subtraction, division, 
remainders, negation, exponentiation, count the members in a DAML 
list, and do the normal truth checking functions, only sub classed 
for numeric values.

Note: see maths with an s for te string-oriented versions.

cf. http://www.w3.org/2000/10/swap/cwm.py and 
http://ilrt.org/discovery/chatlogs/rdfig/2001-12-01.txt from 
"01:20:58" onwards.
"""

__author__ = 'Sean B. Palmer'
__cvsid__ = '$Id: cwm_math.py,v 1.26 2007/06/26 02:36:15 syosi Exp $'
__version__ = '$Revision: 1.26 $'

import sys, string, re, urllib

from term import LightBuiltIn, Function, ReverseFunction, ArgumentNotLiteral, Literal
from local_decimal import Decimal
import types

# from RDFSink import DAML_LISTS, RDF_type_URI, DAML_sameAs_URI

MATH_NS_URI = 'http://www.w3.org/2000/10/swap/math#'

from diag import progress
import sys, traceback

def obsolete():
    progress("Warning: Obsolete math built-in used.")
    traceback.print_stack()

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

def numeric(s):
    if type(s) == types.IntType or \
       type(s) == types.LongType or \
       type(s) is types.FloatType or \
       isinstance(s,Decimal): return s
    if not isinstance(s, (Literal, str, unicode)):
        raise ArgumentNotLiteral(s)
    if s.find('.') < 0 and s.find('e') < 0 : return long(s)
    if 'e' not in s and 'E' not in s: return Decimal(s)
    return float(s)

class BI_absoluteValue(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        return abs(numeric(subj_py))

class BI_rounded(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        return round(float(subj_py))

class BI_sum(LightBuiltIn, Function):
    def evaluateObject(self,  subj_py):
        t = 0
        for x in subj_py: t += numeric(x)
        return t


class BI_sumOf(LightBuiltIn, ReverseFunction):
    def evaluateSubject(self, obj_py): 
        t = 0
        obsolete()
        for x in obj_py: t += numeric(x)
        return t


class BI_difference(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        if len(subj_py) == 2:
            return numeric(subj_py[0]) - numeric(subj_py[1])

class BI_differenceOf(LightBuiltIn, ReverseFunction):
    def evaluateSubject(self,  obj_py): 
        obsolete()
        if len(obj_py) == 2: return numeric(obj_py[0]) - numeric(obj_py[1])


class BI_product(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        t = 1
        for x in subj_py: t *= numeric(x)
        return t

class BI_factors(LightBuiltIn, ReverseFunction):
    def evaluateSubject(self, obj_py): 
        obsolete()
        t = 1
        for x in obj_py: t *= numeric(x)
        return t

class BI_quotient(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        if len(subj_py) == 2:
            if isinstance(numeric(subj_py[0]), long):
                return numeric(subj_py[1]).__rtruediv__(numeric(subj_py[0]))
            return numeric(subj_py[0]).__truediv__(numeric(subj_py[1]))

class BI_integerQuotient(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        if len(subj_py) == 2: return long(subj_py[0]) / long(subj_py[1])

class BI_bit(LightBuiltIn, Function):
    """@@needs a test."""
    def evaluateObject(self, subj_py): 
        if len(subj_py) == 2:
            x = subj_py[0]
            b = subj_py[1]
            return (x >> b) & 1

class BI_quotientOf(LightBuiltIn, ReverseFunction):
    def evaluateSubject(self,  obj_py): 
        obsolete()
        if len(obj_py) == 2: return numeric(obj_py[0]).__truediv__(numeric(obj_py[1]))

# remainderOf and negationOf

class BI_remainder(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        if len(subj_py) == 2: return numeric(subj_py[0]) % numeric(subj_py[1])


class BI_remainderOf(LightBuiltIn, ReverseFunction):
    def evaluateSubject(self,  obj_py): 
        obsolete()
        if len(obj_py) == 2: return numeric(obj_py[0]) % numeric(obj_py[1])

class BI_negation(LightBuiltIn, Function, ReverseFunction):
    def evaluateSubject(self, obj_py): 
            return -numeric(obj_py)
    def evaluateObject(self, subj_py): 
            return -numeric(subj_py)


# Power

class BI_exponentiation(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        if len(subj_py) == 2: return numeric(subj_py[0]) ** numeric(subj_py[1])


class BI_exponentiationOf(LightBuiltIn, ReverseFunction):

    def evaluateSubject(self, obj_py): 
        obsolete()
        if len(obj_py) == 2: return numeric(obj_py[0]) ** numeric(obj_py[1])

# Math greater than and less than etc., modified from cwm_string.py
# These are truth testing things  - Binary logical operators

class BI_greaterThan(LightBuiltIn):
    def evaluate(self, subject, object):
        return (float(subject) > float(object))

class BI_notGreaterThan(LightBuiltIn):
    def evaluate(self, subject, object):
        return (float(subject) <= float(object))

class BI_lessThan(LightBuiltIn):
    def evaluate(self, subject, object):
        return (float(subject) < float(object))

class BI_notLessThan(LightBuiltIn):
    def evaluate(self, subject, object):
        return (float(subject) >= float(object))

class BI_equalTo(LightBuiltIn):
    def evaluate(self, subject, object):
        return (float(subject) == float(object))

class BI_notEqualTo(LightBuiltIn):
    def evaluate(self, subject, object):
        try:
            return (float(subject) != float(object))
        except  (ValueError, AttributeError):
            return None # AttributeError: Symbol instance has no attribute '__float__'
        # or: ValueError: invalid literal for float(): PT1H
            
# memberCount - this is a proper forward function

class BI_memberCount(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        return len(subj_py)

#  Register the string built-ins with the store

def register(store):
    str = store.symbol(MATH_NS_URI[:-1])
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
    str.internFrag('bit', BI_bit)
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
