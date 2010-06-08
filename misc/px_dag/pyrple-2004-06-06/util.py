#!/usr/bin/python
"""util.py - Utilities for Pyrple."""

import random

def label(length=None): 
   """Return one of 183,123,959,522,816 possible labels."""
   alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
   alphanum = alpha + '0123456789'
   first = random.choice(alpha)
   rest = [random.choice(alphanum) for i in xrange(length or 7)]
   new_label = first + ''.join(rest)
   return new_label # @@ store them, and check them?

if __name__=="__main__": 
   print __doc__
