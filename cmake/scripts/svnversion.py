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

def revision(wc_path=None):
  """
  Return the SVN revision of the given path, or None if the given path
  is not a working copy. If path is None, the current working directory is
  queried. Throws IOError if the path doesn't exist.
  """ 
  from os.path import realpath, exists

  if wc_path == None:
    from os import getcwd
    wc_path = getcwd()

  wc_path = realpath(wc_path) 

  if not exists(wc_path):
    from os import strerror
    from errno import errorcode, ENOENT 
    raise IOError(ENOENT, strerror(ENOENT), wc_path) 

  windows = 0
  proc = None

  try:
    from sys import getwindowsversion
    # On Windows, we might have to fall back to subwcrev.exe (from TortoiseSVN). 

    windows = 1

    try:
      proc = process( "svnversion %s"
                    % wc_path.replace('\\', '/'))
    except WindowsError, err:
      # We couldn't find svnversion, fallback to subwcrev.exe
      from tempfile import NamedTemporaryFile
      from os import unlink 
      from os.path import normpath

      # Create two temporary files for use with subwcrev.exe
      input = NamedTemporaryFile(delete=False)
      output = NamedTemporaryFile(delete=False)

      input.write("$WCREV$$WCMODS?M:$\n")

      input.close()
      output.close()

      proc = process( "subwcrev.exe %s %s %s"
                    % ( wc_path.replace('\\', '/')
                      , input.name.replace('\\', '/')
                      , output.name.replace('\\', '/')))

      proc.wait()

      results = open(output.name.replace('\\', '/'))
      raw = results.read().rstrip()
      results.close()

      # Clean up the temporary files
      unlink(input.name.replace('\\', '/'))
      unlink(output.name.replace('\\', '/'))

      if 0 == len(raw):
        return None
      else:
        return raw

  except ImportError, err:
    # On POSIX, svnversion should always be available if SVN is installed.
    proc = process("svnversion %s" % wc_path)

  proc.wait()

  raw = proc.read().rstrip()

  if "Unversioned file" == raw or "Unversioned directory" == raw:
    return None 
  else:
    return raw

from os.path import exists, join

from sys import path, exit

from optparse import OptionParser

usage = "Usage: %prog [options] [path]" 

parser = OptionParser(usage=usage)

(options, wc_path) = parser.parse_args()

if 0 == len(wc_path):
  # If no path is specified, use the current working directory
  print revision()
elif 1 != len(wc_path):
  print "More than one path specified."
  exit(1)
else:
  print revision(wc_path[0])

