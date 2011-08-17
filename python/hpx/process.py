#! /usr/bin/env python 
#
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from threading import Thread

# subprocess instantiation wrapper. Unfortunately older Python still lurks on
# some machines.
try:
  from subprocess import Popen, STDOUT, PIPE
  from types import StringType
  from shlex import split
 
  class process:
    _proc = None
    _exec = None
        
    def __init__(self, cmd):
      if StringType == type(cmd):
        cmd = split(cmd)
      self._proc = Popen(cmd, stderr = STDOUT, stdout = PIPE, shell = False) 

    def poll(self):
      return self._proc.poll()
        
    def pid(self):
      return self._proc.pid

    def _call(self):
      # annoyingly, KeyboardInterrupts are transported to threads, while most
      # other Exceptions aren't in python
      try:
        self._proc.wait()
      except Exception, err:
        self._exec = err
 
    def wait(self, timeout=None):
      if timeout is not None:
        thread = Thread(target=self._call)
        thread.start()

        # wait for the thread and invoked process to finish
        thread.join(timeout)

        # be forceful
        if thread.is_alive():
          self._proc.terminate()
          thread.join()
         
          # if an exception happened, re-raise it here in the master thread 
          if self._exec is not None:
            raise self._exec

          return (True, self._proc.returncode)

        if self._exec is not None:
          raise self._exec

        return (False, self._proc.returncode)

      else:
        return (False, self._proc.wait())

    def read(self):
      return self._proc.stdout.read()

except ImportError, err:
  # no "subprocess"; use older popen module
  from popen2 import Popen4
  from signal import SIGKILL
  from os import kill, waitpid, WNOHANG

  class process:
    _proc = None
    
    def __init__(self, cmd):
      self._proc = Popen4(cmd)

    def poll(self):
      return self._proc.poll()

    def pid(self):
      return self._proc.pid
    
    def _call(self):
      # annoyingly, KeyboardInterrupts are transported to threads, while most
      # other Exceptions aren't in python
      try:
        self._proc.wait()
      except Exception, err:
        self._exec = err

    def wait(self, timeout=None):
      if timeout is not None:
        thread = Thread(target=self._call)
        thread.start()

        # wait for the thread and invoked process to finish
        thread.join(timeout)

        # be forceful
        if thread.is_alive():
          kill(self._proc.pid, SIGKILL)
          waitpid(-1, WNOHANG)
          thread.join()
          
          # if an exception happened, re-raise it here in the master thread 
          if self._exec is not None:
            raise self._exec

          return (True, self._proc.wait())
          
        if self._exec is not None:
          raise self._exec

        return (False, self._proc.wait())

      else:
        return (False, self._proc.wait())

    def read(self):
      return self._proc.fromchild.read()


