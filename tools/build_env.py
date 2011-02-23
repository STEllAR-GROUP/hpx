#! /usr/bin/env python 

"""
=========
build_env
=========

Module for generating a string which identifies a C or C++ development
environment. The identifier string takes the following form::

  processor  ::= [a-z0-9\\-.]+
  kernel     ::= [a-z0-9\\-.]+
  compiler   ::= [a-z0-9\\-.]+
  identifier ::= processor '_' kernel '_' compiler
"""

# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from platform import uname
from re import compile
from string import lower
from types import StringType

try:
  import subprocess
    
  class process:
    _proc = None
        
    def __init__(self, cmd):
      self._proc = subprocess.Popen(cmd,
        stderr = subprocess.STDOUT,
        stdout = subprocess.PIPE,
        shell = (False, True)[type(cmd) == StringType])
            
    def wait(self):
      return self._proc.wait()

    def read(self):
      return self._proc.stdout.read()
            
except ImportError, err:
  # no "subprocess"; use older popen module
  import popen2

  class process:
    _proc = None
    
    def __init__(self, cmd):
      self._proc = popen2.Popen4(cmd)

    def wait(self):
      return self._proc.wait()

    def read(self):
      return self._proc.fromchild.read()

def make_component(raw, type):
  comp = compile(r'\s|_').sub('-', lower(raw))

  if compile(r'^[a-z0-9\-.]+$').match(comp):
    return comp
  else:
    return "unknown-%s" % type

def make_compiler_component(driver):
  windows = 0

  try:
    from sys import getwindowsversion
    # on windows, running cl.exe with no args returns what we want
    windows = 1
    proc = process("%s" % driver)
  except ImportError, err:
    # on POSIX, assume GNU-style long options
    proc = process("%s --version" % driver)

  proc.wait()
  raw = proc.read() 

  if (windows):
    compiler = compile(r'Version ([0-9.]+)').match(raw) 
 
    if (compiler):
      compiler = compiler.expand(r'msvc-\2')
      if compile(r'^[a-z0-9\-.]+$').match(compiler):
        return compiler
      else:
        return "msvc" 

  # handle GNU GCC and Intel
  compiler = compile(r'^(icc|icpc|gcc|g[+][+]) [(][^)]+[)] ([0-9.]+)').match(raw)
  
  if (compiler):
    unescaped = compiler.expand(r'\1-\2')
    compiler = compile(r'[+]').sub("x", unescaped)

    if compile(r'^[a-z0-9\-.]+$').match(compiler):
      return compiler
    else:
      unescaped = compile(r'^(icc|icpc|gcc|g[+][+])').match(raw).expand(r'\1')
      return compile(r'[+]').sub("x", unescaped)
 
  # handle Clang
  compiler = compile(r'(clang) version ([0-9.]+)').match(raw)
  
  if (compiler):
    compiler = compiler.expand(r'\1-\2')
    if compile(r'^[a-z0-9\-.]+$').match(compiler):
      return compiler
    else:
      return compile(r'^(clang)').match(raw).expand(r'\1')

  return "unknown-compiler"

def identify(driver):
  (system, node, release, version, machine, processor) = uname()

  if len(processor) == 0:
    processor = machine

  return "%s_%s-%s_%s" % (make_component(processor, "processor"),
                          make_component(system, "kernel"),
                          make_component(release, "version"),
                          make_compiler_component(driver)) 

if __name__ == "__main__":
  from sys import argv 
  command = ""
  for arg in argv[1:]:
    command += ("\"%s\"" % arg)
  print identify(command)

