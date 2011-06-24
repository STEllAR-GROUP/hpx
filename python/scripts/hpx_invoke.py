#! /usr/bin/env python
#
# Copyright (c) 2009 Maciej Brodowicz
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from os.path import exists, join

from sys import path

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
  start = datetime.now() 
  proc = process(cmd)
  (timed_out, returncode) = proc.wait(timeout)
  now = datetime.now()

  output = ''

  while True:
    s = proc.read()

    if s:
      output += s
    else:
      break

  return (returncode, output, timed_out)

def quote_command(cmd, quote = '"'):
  exclude = letters + digits + '-+='
  s = ''
  for e in cmd:
    if type(e) is not StringType:
      e = str(e)
    for c in e:
      if c not in exclude:
        s += ' ' + quote + e + quote
        break
      else:
        s += ' ' + e
  return s

def rstrip_last(s, chars):
  if s[-1] in chars:
    return s[:-1]
  else:
    return s

# {{{ main
usage = "usage: %prog [options] program [program-arguments]" 

parser = OptionParser(usage=usage)

parser.add_option("--verbatim",
                  action="store_true",
                  dest="verbatim", default=False,
                  help="Don't quote program arguments")

parser.add_option("--timeout",
                  action="store", type="int",
                  dest="timeout", default=3600,
                  help="Program timeout (seconds)")

(options, cmd) = parser.parse_args()

if options.verbatim:
  cmd = cmd[0] + quote_command(cmd[1:], '')
else:
  cmd = cmd[0] + quote_command(cmd[1:])

(returncode, output, timed_out) = run(cmd, options.timeout)

if not 0 == len(output):
  print rstrip_last(output, '\n')

if timed_out:
  print "Program timed out"
  
exit(returncode)
# }}}

