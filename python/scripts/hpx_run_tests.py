#! /usr/bin/env python
#
# Copyright (c) 2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# TODO: Fractional threads_per_locality

import sys, os, string
import os.path as osp

from types import StringType

from optparse import OptionParser

from errno import ENOENT

import signal, re

import subprocess

if __name__ == '__main__':
  # {{{ main
    usage = "Usage: %prog [ctest option files]"

    parser = OptionParser(usage=usage)
    parser.add_option("--prefix",
                    action="store", type="string",
                    dest="prefix", default="",
                    help="")

    parser.add_option("--log-stdout",
                    action="store_true", dest="log_stdout", default=False,
                    help="Send logs to stdout (overrides --log-prefix)")

    (ignore, files) = parser.parse_args();
    base_cmd = "ctest --output-on-failure "

    options = "";
    for f in files:
        options += eval(open(f).read())

    cmd = base_cmd + options
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    result, out = p.communicate()

    exit(result)
  # }}}

