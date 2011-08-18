#! /usr/bin/env python 
#
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from os.path import exists, join

from sys import path, exit

from optparse import OptionParser

if exists(join(path[0], "../hpx")):
  path.append(join(path[0], ".."))
if exists(join(path[0], "../share/hpx/python/hpx")):
  path.append(join(path[0], "../share/hpx/python"))

from hpx.environment import identify

usage = "Usage: %prog [options] compiler-driver" 

parser = OptionParser(usage=usage)

(options, driver) = parser.parse_args()

if 0 == len(driver):
  print "No compiler driver specified."
  exit(1) 
elif 1 != len(driver):
  print "More than one compiler driver specified."
  exit(1)

print identify(driver[0])

