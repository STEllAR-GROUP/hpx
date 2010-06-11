"""Decimal datatype

This is an implementation of the Decimal XML schema datatype in python

magnitude is the log10 of the number we multiply it by to get an integer

$Id: local_decimal.py,v 1.2 2006/01/10 13:58:47 syosi Exp $
"""

from types import IntType, FloatType, LongType, StringTypes
from math import log10

class Decimal:
    """make a new Decimal

    Argument can be string, int, long, or float
    float is not recommended
    """

    _limit = 16
    
    def normalize(self):
        """convert this Decimal into some sort of canonical form


        """
        if self.value == 0:
            self.magnitude = 0
            return
        while self.value.__mod__(10) == 0:
            self.value = self.value / 10
            self.magnitude = self.magnitude - 1
##        while self.magnitude > 2 * self.__class__._limit:
##            self.value = self.value / 10
##            self.magnitude = self.magnitude - 1
        
        
    
    def __init__(self, other=0):
        """How to get a new Decimal

        What happened?
        """
        if isinstance(other, Decimal):
            self.value = other.value
            self.magnitude = other.magnitude
            return
        elif isinstance(other, IntType):
            self.value = long(other)
            self.magnitude = 0
            self.normalize()
            return
        elif isinstance(other, LongType):
            self.value = other
            self.magnitude = 0
            self.normalize()
            return
        elif hasattr(other,'__Decimal__') and callable(getattr(other, '__Decimal__')):
            a = other.__Decimal__()
            self.value = a.value
            self.magnitude = a.magnitude
            self.normalize()
            return
        elif isinstance(other,FloatType):
            other = `other`
        try:
            other[0]
        except TypeError:
            other = `other`
        other = other + 'q'
        i = 0
        value = long(0)
        magnitude = long(0)
        sign = 1
        newsign = 1
        base = 10
        magnitude_multiplier = 0
        if other[i] == '-':
            sign = -1
            i = i + 1
        while other[i] in '0123456789':
            ours = other[i]
            i = i+1
            value = value*base+int(ours, base)
        if other[i] == '.':
            i = i+1
            while other[i] in '0123456789':
                ours = other[i]
                i = i+1
                value = value*base+int(ours, base)
                magnitude = magnitude + 1
        if other[i] in 'eE':
            i = i+1
            if other[i] == '+':
                i = i+1
            elif other[i] == '-':
                newsign = -1
                i = i+1
            while other[i] in '0123456789':
                ours = other[i]
                i = i+1
                magnitude_multiplier = magnitude_multiplier*10+int(ours, 10)
        self.magnitude = magnitude-newsign*magnitude_multiplier
        self.value = value*sign
        self.normalize()

        
    def __abs__(self):
        """x.__abs__() <==> abs(x)
        """
        a = self.__class__(self)
        a.value = abs(a.value)
        a.normalize()
        return a
    def __add__(self, other):
        """x.__add__(y) <==> x+y
        """
#        if not isinstance(other, Decimal):
#            other = self.__class__(other)
        if other.magnitude < self.magnitude:
            return other.__add__(self)
        while other.magnitude > self.magnitude:
            self.magnitude = self.magnitude+1
            self.value = self.value * 10
        a = self.__class__()
        a.value = self.value + other.value
        a.magnitude = self.magnitude
        self.normalize()
        a.normalize()
        return a
    def __cmp__(self, other):
        """x.__cmp__(y) <==> cmp(x,y)
        """
        if not isinstance(other, Decimal):
            other = self.__class__(other)
        if other.magnitude < self.magnitude:
            return -other.__cmp__(self)
        while other.magnitude > self.magnitude:
            self.magnitude = self.magnitude+1
            self.value = self.value * 10
        a = cmp(self.value, other.value)
        self.normalize()
        return a
    def __coerce__(self, other):
        """x.__coerce__(y) <==> coerce(x, y)
        """
        if other.__class__ == float:
            return float(self), other
        return self, self.__class__(other)
        pass
    def __div__(self, other):
        """x.__div__(y) <==> x/y
        """
        while self.magnitude <  self.__class__._limit + other.magnitude + int(log10(other)):
            self.value = self.value * 10
            self.magnitude = self.magnitude + 1
        if self.value % other.value:
            a = float(self) / float(other)
        else:
            a = self.__class__()
            a.value = self.value // other.value
            a.magnitude = self.magnitude - other.magnitude
            a.normalize()
        self.normalize()
        if a == NotImplemented:
            raise RuntimeError
        return a
    def __divmod__(self, other):
        """x.__divmod__(y) <==> divmod(x, y)
        """
        return (self // other, self % other)
    def __float__(self):
        """x.__float__() <==> float(x)
        """
        return float(self.value * 10**(-self.magnitude))
    def __floordiv__(self, other):
        """x.__floordiv__(y) <==> x//y
        """
#        if not isinstance(other, Decimal):
#            other = self.__class__(other)
        if other.magnitude < self.magnitude:
            return other.__rfloordiv__(self)
        while other.magnitude > self.magnitude:
            self.magnitude = self.magnitude+1
            self.value = self.value * 10
        a = self.__class__()
        a.magnitude = 0
        a.value = self.value // other.value
        a.normalize()
        return a
        
    def __hash__(self):
        """x.__hash__() <==> hash(x)
        """
        return hash((self.value, self.magnitude))
    def __int__(self):
        """x.__int__() <==> int(x)
        """
        value = self.value
        power = self.magnitude
        while power > 0:
            value = value // 10
            power = power - 1
        return int(value * 10**(-power))
    def __long__(self):
        """x.__long__() <==> long(x)
        """
        value = self.value
        power = self.magnitude
        while power > 0:
            value = value // 10
            power = power - 1
        return long(value * 10**(-power))
    def __mod__(self, other):
        """x.__mod__(y) <==> x%y
        """
#        if not isinstance(other, Decimal):
#            other = self.__class__(other)
        if other.magnitude < self.magnitude:
            return other.__rmod__(self)
        while other.magnitude > self.magnitude:
            self.magnitude = self.magnitude+1
            self.value = self.value * 10
        a = self.__class__()
        a.magnitude = self.magnitude
        a.value = self.value % other.value
        a.normalize()
        return a
    def __mul__(self, other):
        """x.__mul__(y) <==> x*y
        """
#        if not isinstance(other, Decimal):
#            other = self.__class__(other)
        a = self.__class__()
        a.value = self.value * other.value
        a.magnitude = self.magnitude + other.magnitude
        a.normalize()
        return a
    def __neg__(self):
        """x.__neg__ <==> -x
        """
        a = self.__class__(self)
        a.value = -a.value
        return a
    def __nonzero__(self, other):
        """x.__nonzero__() <==> x != 0
        """
        return self.value != 0
    def __pos__(self):
        """x.__pos__() <==> +x
        """
        return self.__class__(self)
    def __pow__(self, other, mod=0):
        """x.__pow__(y[, z]) <==> pow(x, y[, z])

        If the exponent is not an integer, we will simply convert things to floats first
        """
        if not isinstance(other, Decimal):
            other = self.__class__(other)
        while other.magnitude < 0:
            other.value = other.value*10
            other.magnitude = other.magnitude + 1
        if other.magnitude == 0:
            a = self.__class__()
            a.value = self.value ** other.value
            a.magnitude = self.magnitude * other.value
            a.normalize()
            if mod !=0:
                a = a%mod
            return a
        else:
    #I honestly think that here we can give up on accuracy
##            tempval = self.__class__(self.value ** other.value)
##            tempval2 = self.__class__(10 ** (self.magnitude * other.value))
##            temppow = 10 ** other.magnitude
##            a = self.__class__(n_root(tempval, temppow))
##            b = self.__class__(n_root(tempval2, temppow))
##            c = a / b
##            c.normalize()
            a = self.__class__(pow(float(self),float(other),mod))
            return a
        
    def __radd__(self, other):
        """x.__radd__(y) <==> y+x
        """
        return self.__add__(other)
    def __rdiv__(self, other):
        """x.__rdiv__(y) <==> y/x
        """
        if not isinstance(other, Decimal):
            other = self.__class__(other)
        return other.__div__(self)
    def __rdivmod__(self, other):
        """x.__rdivmod__(y) <==> divmod(y, x)
        """
        return other.__rdivmod__(self)
    def __repr__(self):
        """x.__repr__() <==> repr(x)
        """
        return '%s("%s")' % (self.__class__.__name__, str(self))
    def __rfloordiv__(self, other):
        """x.__rfloordiv__(y) <==> y//x
        """
#        if not isinstance(other, Decimal):
#            other = self.__class__(other)
        if other.magnitude < self.magnitude:
            return other.__floordiv__(self)
        while other.magnitude > self.magnitude:
            self.magnitude = self.magnitude+1
            self.value = self.value * 10
        a = self.__class__()
        a.magnitude = 0
        a.value = other.value // self.value
        a.normalize()
        return a
    def __rmod__(self, other):
        """x.__rmod__(y) <==> y%x
        """
        if not isinstance(other, Decimal):
            other = self.__class__(other)
        if other.magnitude < self.magnitude:
            return other.__mod__(self)
        while other.magnitude > self.magnitude:
            self.magnitude = self.magnitude+1
            self.value = self.value * 10
        a = self.__class__()
        a.magnitude = self.magnitude
        a.value = other.value % self.value
        a.normalize()
        return a
    def __rmul__(self, other):
        """x.__rmul__(y) <==> y*x
        """
        return self.__mul__(other)
    def __rpow__(self, other, mod=0):
        """y.__rpow__(x[, z]) <==> pow(x, y[, z])
        """
        return other.__pow__(self, mod)
    def __rsub__(self, other):
        """x.__rsub__(y) <==> y-x
        """
        a = self.__class__(self)
        a.value = -a.value
        return a.__add__(other)
    def __rtruediv__(self, other):
        """x.__rtruediv__(y) <==> y/x
        """
        return self.__rdiv__(other)
#    def __setattr__(self, other):
#        pass
    def __str__(self):
        """x.__str__() <==> str(x)
        """
        magnitude = self.magnitude
        value = self.value
        output = []
        if value == 0:
            return "0"
        try:
            magSign = magnitude / abs(magnitude)
        except ZeroDivisionError:
            magSign = 0
        sign = value / abs(value)
        value = abs(value)
        while magnitude < 0:
            output.append("0")
            magnitude = magnitude + 1
        while value != 0:
            if magnitude == 0 and magSign == 1:
                output.append(".")
                magSign = 0
            digit = value.__mod__(10)
            value = value // 10
            output.append("0123456789"[digit])
            magnitude = magnitude-1
        while magnitude > 0:
            output.append("0")
            magnitude = magnitude - 1
        if magSign == 1:
            output.append('0.')
        if sign == -1:
            output.append('-')
        output.reverse()
        return "".join(output)
    def __sub__(self, other):
        """x.__sub__(y) <==> x-y
        """
        a = self.__class__(other)
        a.value = -a.value
        return self.__add__(a)
    def __truediv__(self, other):
        """x.__truediv__(y) <==> x/y
        """
        return self.__div__(self.__class__(other))

def n_root(base, power):
    """Find the nth root of a Decimal
    """
    print 'trying to compute ', base, ' ** 1/ ', power
    accuracy = Decimal(1)
    n = 10 #Decimal._limit
    while n > 0:
        accuracy = accuracy / 10
        n = n-1
    oldguess = Decimal(0)
    guess = Decimal('.00000002')
    counter = 0
    while 1:
        oldguess = guess
        counter = counter + 1
        if counter == 100:
            print guess
            counter = 0
        h = 1 - base * (guess ** power)
        guess = guess + guess * h / power
        if abs(guess - oldguess) <= accuracy:
#            print guess
            break
#    print guess
    answer = Decimal(1) / Decimal(guess)
    print answer
    return answer
