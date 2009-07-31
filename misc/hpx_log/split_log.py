#!/usr/bin/python
"""split_log.py - HPX log splitter. Generates new log files <log-filename>.<locality>

\tusage: python split_log.py <log-filename>
"""

# Copyright (c) 2009-2010 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import sys

from hpx_log import HpxLog

def run(log_filename):
  log = HpxLog(log_filename)
  print "Processing %s" % (log.get_filename())

  locality = {}
  for event in log.get_events():
    if locality.has_key(event['locality']):
      locality[event['locality']].append(str(event))
    else:
      locality[event['locality']] = [str(event)]

  for (locality, lines) in locality.items():
    filename = "%s.%s" % (log.get_filename(), locality)
    file = open(filename, 'w')

    lines.sort()
    for line in lines:
      file.write(line+'\n')

if __name__=="__main__":
  if (len(sys.argv) == 2):
    log_file = sys.argv[1]
    run(log_file)
  else:
    print __doc__;
