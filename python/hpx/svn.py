#! /usr/bin/env python 
#
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

"""
Module for determining the SVN revision of a working copy. Works with symlinks. 
"""

def revision(wc_path=None):
  """
  Return the SVN revision of the given path, or None if the given path
  is not a working copy. If path is None, the current working directory is
  queried. Throws IOError if the path doesn't exist.
  """ 
  from hpx.process import process
  from os.path import realpath, exists

  if wc_path == None:
    from os import getcwd
    wc_path = getcwd()

  wc_path = realpath(wc_path) 

  if not exists(wc_path):
    from os import strerror
    from errno import errorcode, ENOENT 
    raise IOError(ENOENT, strerror(ENOENT), wc_path) 

  proc = process("svnversion %s" % wc_path)
  proc.wait()

  raw = proc.read().rstrip()

  if "Unversioned file" == raw or "Unversioned directory" == raw:
    return None 
  else:
    return raw

