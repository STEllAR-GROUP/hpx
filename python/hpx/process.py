#! /usr/bin/env python 
#
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# subprocess instantiation wrapper. Unfortunately older Python still lurks on
# some machines.
try:
  from subprocess import Popen, STDOUT, PIPE
  from types import StringType
    
  class process:
    _proc = None
        
    def __init__(self, cmd):
      self._proc = Popen(cmd, stderr = STDOUT, stdout = PIPE,
        shell = (False, True)[type(cmd) == StringType])

    def poll(self):
      return self._proc.poll()
        
    def pid(self):
      return self._proc.pid
            
    def wait(self):
      return self._proc.wait()

    def read(self):
      return self._proc.stdout.read()
            
except ImportError, err:
  # no "subprocess"; use older popen module
  from popen2 import Popen4

  class process:
    _proc = None
    
    def __init__(self, cmd):
      self._proc = Popen4(cmd)

    def poll(self):
      return self._proc.poll()

    def pid(self):
      return self._proc.pid

    def wait(self):
      return self._proc.wait()

    def read(self):
      return self._proc.fromchild.read()


