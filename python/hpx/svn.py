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

