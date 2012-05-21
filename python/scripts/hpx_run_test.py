#! /usr/bin/env python
#
# Copyright (c) 2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# TODO: Fractional threads_per_locality

import sys, os, string

from types import StringType

from optparse import OptionParser

if os.path.exists(os.path.join(sys.path[0], "../hpx")):
  sys.path.append(os.path.join(sys.path[0], ".."))
if os.path.exists(os.path.join(sys.path[0], "../share/hpx/python/hpx")):
  sys.path.append(os.path.join(sys.path[0], "../share/hpx/python"))

from hpx.process import process, process_group

# Input files should be a list of lists. Each sublist should follow the
# following structure:
#
# format: [ name, timeout, success, nodes, threads_per_node, args ] 
# types:  [ string, float or None, bool, int, int, list ]

def create_path(name, prefix="", suffix=""):
  return os.path.expandvars(prefix + name + suffix)

def quote_options(options, quoting_char = '"'):
  no_quote = string.letters + string.digits + '-+=/_'
  s = ''

  for option in options:
    if type(option) is not StringType:
      option = str(option)
    for c in option:
      if c not in no_quote:
        s += ' ' + quoting_char + option + quoting_char
        break
    else:
      s += ' ' + option

  return string.strip(s)

if __name__ == '__main__':
  # {{{ main
  usage = "Usage: %prog [options] [.tests files]" 

  parser = OptionParser(usage=usage)

  parser.add_option("--suffix",
                    action="store", type="string",
                    dest="suffix", default="_test",
                    help="Suffix added to test names [default: %default]") 

  parser.add_option("--prefix",
                    action="store", type="string",
                    dest="prefix", default="./",
                    help="Prefix added to test names") 

  parser.add_option("--log",
                    action="store", type="string",
                    dest="log", default="fail",
                    help="Always log output (--log=always), never log "
                        +"output (--log=never) or log output for tests "
                        +"that fail (--log=fail) [default: %default]")

  parser.add_option("--log-prefix",
                    action="store", type="string",
                    dest="log_prefix", default="./",
                    help="Prefix for log files") 

  (options, files) = parser.parse_args()

  if not (lambda x: "always" == x or "never" == x or "fail" == x)(options.log):
    print "Error: --log=" + quote_options([options.log]) + " is invalid\n"
    parser.print_help() 
    sys.exit(1) 

  if 0 == len(files):
    print "Error: no .tests files specified\n" 
    parser.print_help() 
    sys.exit(1) 

  tests = []

  for f in files:
    tests += eval(open(f).read())

  for [name, timeout, success, nodes, threads_per_node, args] in tests:
    full_name = create_path(name, options.prefix, options.suffix)

    print "Running: " + full_name

    pg = process_group()

    cmds = {}

    for node in range(nodes):
      cmd = quote_options([ full_name 
                          , '-t' + str(threads_per_node)
                          , '-l' + str(nodes)
                          , '-' + str(node)] + args)

      cmds[pg.create_process(cmd).fileno()] = (node, cmd) 

    def print_result(fd, job, output):
      passed = (job.poll() == 0 if success else job.poll() != 0)

      print (" " * 2) + cmds[job.fileno()][1]
      print (" " * 4) + "Result: " + ("Passed" if passed else "Failed")
      print (" " * 4) + "Exit code:", job.poll()
      print (" " * 4) + "Timed out:", job.timed_out()

      if "always" == options.log or ("fail" == options.log and not passed):
        if 0 != len(output):
          log = create_path(name, options.log_prefix, options.suffix) \
              + "_l" + str(cmds[job.fileno()][0]) + ".log"
          print >> open(log, "w"), output,
          print (" " * 4) + "Output log: " + log

    pg.read_all(timeout, print_result)
  # }}}

