"""Trigonometrical Built-Ins for CWM

Allows CWM to do do trigonometrical
http://www.python.org/doc/2.3/lib/module-math.html

This module is inspired by the math module.
See http://www.w3.org/2000/10/swap/cwm_math.py

cf. http://www.w3.org/2000/10/swap/cwm.py
See http://ilrt.org/discovery/chatlogs/rdfig/2003-09-23.html#T22-37-54
http://rdfig.xmlhack.com/2003/09/23/2003-09-23.html#1064356689.846120


"""

__author__ = 'Karl Dubost'
__cvsid__ = '$Id: cwm_trigo.py,v 1.13 2007/06/26 02:36:15 syosi Exp $'
__version__ = '$Revision: 1.13 $'
__all__ = ["evaluateObject"]

from math import sin, acos, asin, atan, atan2, cos, cosh, sinh, tan, tanh
from term import LightBuiltIn, Function, ReverseFunction
import types
from diag import progress
from cwm_math import *

MATH_NS_URI = 'http://www.w3.org/2000/10/swap/math#'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Trigonometrical Features
#
#
# Light Built-in classes - these are all reverse functions
#
#  Note that asin the arc sine i s just the reverse function of sin,
#  handled by sin being a reverse function as well as a function
# cosine,  hyperbolic or not
# sine,  hyperbolic or not
# tangent, arc tangent, arc tangent (y/x)
#


class BI_atan2(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        """atan2(y, x)
        
        Return the arc tangent (measured in radians) of y/x.
        Unlike atan(y/x), the signs of both x and y are considered.
        -- Karl""" 
        if len(numeric(subj_py)) == 2:
                return atan2(numeric(subj_py[0]),numeric(subj_py[1]))
        else: return None

class BI_cos(LightBuiltIn, Function, ReverseFunction):
    def evaluateObject(self, subj_py):
        """cos(x)
        
        Return the cosine of x (measured in radians)."""
        return cos(numeric(subj_py))

    def evaluateSubject(self, x):
        try:
            return acos(numeric(x))
        except ValueError:
            return None

class BI_cosh(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        """cosh(x)
        
        Return the hyperbolic cosine of x."""
        return cosh(numeric(subj_py))

#    def evaluateSubject(self, x):
#        return acosh(numeric(x))

class BI_degrees(LightBuiltIn, Function, ReverseFunction):
    """Angles are in radians.  This property is the equivalent in degrees.
    It can be calculated either way."""
    def evaluateObject(self, subj_py):
        """Angles are in radians.  This property is the equivalent in degrees."""
        return numeric(subj_py) * 180 / 3.14159265358979323846
    def evaluateSubject(self, obj_py): 
        """radians(x) -> converts angle x from degrees to radian"""
        return numeric(obj_py) * 3.14159265358979323846 / 180

class BI_sin(LightBuiltIn, Function, ReverseFunction):
    """sin(x)
        
    x.math:sin is the sine of x (x measured in radians)."""
    def evaluateObject(self, subj_py):
        return sin(numeric(subj_py))

    def evaluateSubject(self, x):
        try:
            return asin(numeric(x))
        except:
            return None

class BI_sinh(LightBuiltIn, Function):
    def evaluateObject(self, subj_py):
        """sinh(x)
        
        Return the hyperbolic sine of x."""
        return sinh(numeric(subj_py))

#    def evaluateSubject(self, x):
#        return asinh(numeric(x))

class BI_tan(LightBuiltIn, Function, ReverseFunction):
    def evaluateObject(self, subj_py):
        """tan(x)
        
        Return the tangent of x (measured in radians)."""
        return tan(numeric(subj_py))

    def evaluateSubject(self, x):
        """tan(x)
        
        Return the tangent of x (measured in radians)."""
        return atan(numeric(x))

class BI_tanh(LightBuiltIn, Function):
    """tanh(x)
        
        Return the hyperbolic tangent of x."""
    def evaluateObject(self, subj_py):
        return tanh(numeric(subj_py))

#    def evaluateSubject(self, x):
#        return atanh(numeric(x))

#  Register the string built-ins with the store

def register(store):
    str = store.symbol(MATH_NS_URI[:-1])
    str.internFrag('cos', BI_cos)
    str.internFrag('cosh', BI_cosh)
    str.internFrag('degrees', BI_degrees)
    str.internFrag('sin', BI_sin)
    str.internFrag('sinh', BI_sinh)
    str.internFrag('tan', BI_tan)
    str.internFrag('tanh', BI_tanh)
 
if __name__=="__main__": 
   print __doc__.strip()
