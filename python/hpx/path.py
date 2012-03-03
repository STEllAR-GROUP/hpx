#! /usr/bin/env python 
#
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

def absolute_path(p):
  """
  Find the absolute location of a binary in the current path.
  """
  from os import pathsep, environ
  from os.path import isfile, join, realpath
  for dirname in environ['PATH'].split(pathsep):
    candidate = join(dirname, p)
    if isfile(candidate):
      return realpath(candidate)
  return realpath(p) 

