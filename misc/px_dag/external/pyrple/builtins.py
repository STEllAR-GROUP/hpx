#!/usr/bin/env python
"""
builtins.py - Builtins module for pyrple.
By Sean B. Palmer, <http://purl.org/net/sbp>.
GPL 2; share and enjoy!
"""

import sys, os, copy
from node import Var
from triple import Triple

BUG = 0

builtins = {}
try: import extensions
except ImportError: 
   print >> sys.stderr, "No extension modules directory"
else: 
   path = extensions.__path__[0]
   filenames = [fn[:-3] for fn in os.listdir(path) if fn.endswith('.py')]
   for mod in filenames: 
      if not mod.startswith('_'): 
         args = ('extensions', globals(), locals(), [mod])
         builtins.update(getattr(__import__(*args), mod).builtins)
   del extensions

def getBuiltins(ante): 
   ante = ante.copy() # @@ necessary?
   bi = filter(lambda t: builtins.has_key(t[1]), ante)
   for triple in bi: 
      ante.remove(triple)
   if BUG: print 'ante, bi:::', ante, bi
   return ante, bi

class Builtin(object): 
   def __init__(self, strength, type, val): 
      self.strength, self.type, self.val = strength, type, val

   def __repr__(self): 
      return '%r::%r::%r' % (self.strength, self.type, self.val)

def doBuiltins(cons, bi, rdict): 
   if len(bi) == 0: return cons
   bi = [list(t.copy()) for t in bi] # important!

   if BUG: print >> sys.stderr, 'rdict::', rdict
   for i in range(len(bi)): 
      for pos in xrange(3): 
         if type(bi[i][pos]) == Var: 
            if rdict.has_key(bi[i][pos]) and rdict[bi[i][pos]]: 
               bi[i][pos] = rdict[bi[i][pos]]
            elif not rdict.has_key(bi[i][pos]): 
               rdict[bi[i][pos]] = None
   if BUG: print >> sys.stderr, 'bi, rdict::', bi, rdict

   # order by builtin strength
   queue = [Builtin(builtins[b[1]].strength, 
               filter(lambda fun: fun.startswith('do'), 
               dir(builtins[b[1]])), b) for b in bi]
   done = []
   queue.sort(lambda x, y: (x.strength < y.strength) - 1)
   if BUG: print >> sys.stderr, 'queue::', queue

   while 1: 
      queue_length, done_length = len(queue), len(done)
      for todo in queue: 
         for pos in xrange(3): 
            if (type(todo.val[pos]) == Var) and rdict[todo.val[pos]]: 
               todo.val[pos] = rdict[todo.val[pos]]

         if BUG: print >> sys.stderr, 'doing::: %r %r' % (todo.type, todo.val)
         if BUG: print >> sys.stderr, type(todo.val[1]), type(todo.val[2])
         if (('doSubject' in todo.type) # @@ types?
             and (type(todo.val[0]) == Var) 
             and (type(todo.val[2]) != Var)): 
            result = builtins[todo.val[1]]().doSubject(todo.val)
         elif (('doObject' in todo.type) 
             and (type(todo.val[0]) != Var) 
             and (type(todo.val[2]) == Var)): 
            result = builtins[todo.val[1]]().doObject(todo.val)
         elif (('do' in todo.type) 
             and (type(todo.val[0]) != Var) 
             and (type(todo.val[2]) != Var)): 
            if builtins[todo.val[1]]().do(todo.val): result = 1
            else: result = 0 # fail
         else: result = 'WAIT'

         if BUG: print >> sys.stderr, 'result::: %r' % result

         if not result: return []
         elif result == 'WAIT': pass
         elif result == 1: 
            if BUG: print >> sys.stderr, "match:", todo.type, todo.val
            queue.remove(todo)
            done.append(1)
         elif result: 
            if BUG: print >> sys.stderr, "match:", todo.type, todo.val, result
            if type(result) is type([]): 
               result = (result, [])
            for pos in xrange(3): 
               if type(todo.val[pos]) == Var: 
                  rdict[todo.val[pos]] = result[0][pos]
            queue.remove(todo)
            done.append(Triple(result[0]))
            if result[1]: done.extend(result[1])

      if (len(queue) == 0) or (not len(done) > done_length): break

   if not done: return []
   done = filter(lambda n: n != 1, done)
   for r in cons: done.append(Triple([rdict.get(t) or t for t in r]))
   if BUG: print >> sys.stderr, 'done!:::', queue, done
   return cons

if __name__=="__main__": 
   print __doc__
