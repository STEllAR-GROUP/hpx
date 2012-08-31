#! /usr/bin/env python
#
# Copyright (c) 2009 Maciej Brodowicz
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from os.path import exists, join

from datetime import datetime

from string import letters, digits

from types import StringType

from optparse import OptionParser

if exists(join(path[0], "../hpx")):
  path.append(join(path[0], ".."))
if exists(join(path[0], "../share/hpx/python/hpx")):
  path.append(join(path[0], "../share/hpx/python"))

from hpx.process import process

def run(cmd, timeout=3600):
  proc = process(cmd)
  (timed_out, returncode) = proc.wait(timeout)

  output = ''

  while True:
    s = proc.read()

    if s:
      output += s
    else:
      break

  return (timed_out, returncode, output)

def rstrip_last(s, chars):
  if s[-1] in chars:
    return s[:-1]
  else:
    return s

# {{{ main
usage = "Usage: %prog [options]" 

parser = OptionParser(usage=usage)

parser.add_option("--timeout",
                  action="store", type="int",
                  dest="timeout", default=3600,
                  help="Program timeout (seconds)")

parser.add_option("--program",
                  action="store", type="string",
                  dest="program",
                  help="Program to invoke") 

(options, cmd) = parser.parse_args()

if None == options.program:
  print "No program specified"
  exit(1)

(timed_out, returncode, output) = run(options.program, options.timeout)

if not 0 == len(output):
  print rstrip_last(output, '\n')

if timed_out:
  print "Program timed out"
  
exit(returncode)
# }}}

