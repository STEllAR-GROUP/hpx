"""mixin.py

$ID:    $

We want to be able to mixin query functions
These include:

substitution(self, bindings, why=None):
substituteEquals(self, bindings, newRedirections):
occurringIn(self, vars):
unify(self, other, vars, existentials,  bindings):

we want the interface to be convenient.

Here is how to use it:

Let us say, in one file, you have a class foo

class foo(object):
    def something(self):
        ...

and you want to add a other() method to it. Create another file,
and do as follows

------
from mixin import Mixin

class betterFoo(Mixin, foo):
    def other(self):
        ....

----
and import that file where you need to use betterFoo, even AFTER the
objects are created!
"""

##try:
##    Set = set
##except NameError:
##    from sets import Set
##
##operations = {}

class mixinClass(object):
    """

    """
    def __new__(metacls, name, bases, dict):
        for base in bases:
            #print base
            if base.__class__ is not mixinClass:
                for func in dict:
                    #print func
                    if func[0:1] != '_':
##                        if func not in operations:
##                            operations[func] = {}
##                        operations[func][base] = dict[func]
                        if func in base.__dict__:
                            raise ValueError('''I can't let you override an existing method.
Use real inheritance: %s.%s''' % (`base`, func))
                        setattr(base, func, dict[func])
        return object.__new__(metacls)

class Mixin:
    __metaclass__ = mixinClass


                        
