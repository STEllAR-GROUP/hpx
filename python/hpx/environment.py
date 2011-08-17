#! /usr/bin/env python 
#
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
Module for generating strings which describe a C or C++ development environment.
The identifier string takes the following form:::

  processor  ::= [a-z0-9\\-.]+
  kernel     ::= [a-z0-9\\-.]+
  compiler   ::= [a-z0-9\\-.]+
  identifier ::= processor '_' kernel '_' compiler
"""

from re import compile

def make_component(raw, type):
  """
  Transliterate characters from an element returned by platform.uname() to
  match the regex ``^[a-z0-9\-.]+$``. Returns ``"unknown-%s" % type`` if 
  the transliterated string doesn't match the regex pattern. 
  """ 
  from string import lower

  comp = compile(r'\s|_').sub('-', lower(raw))

  if compile(r'^[a-z0-9\-.]+$').match(comp):
    return comp
  else:
    return "unknown-%s" % type

def make_compiler_component(driver):
  """
  Given the name of a compiler driver, generate a string describing the compiler
  suite and version. Returns ``"unknown-compiler"`` if the compiler cannot be
  identified.
  """ 
  from hpx.process import process

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
  compiler = compile(r'^(icc|icpc|gcc|g[+][+])[^ ]* [(][^)]+[)] ([0-9.]+)').match(raw)
  
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
  """
  Given the name of a compiler driver in the current path (or an absolute path
  to a compiler driver), build a complete environment identifier using the
  information provided by platform.uname().
  """
  from platform import uname
  from hpx.path import absolute_path

  (system, node, release, version, machine, processor) = uname()

  return "%s_%s-%s_%s" % (make_component(machine, "processor"),
                          make_component(system, "kernel"),
                          make_component(release, "version"),
                          make_compiler_component(absolute_path(driver))) 

