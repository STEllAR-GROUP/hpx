#!/usr/bin/python
"""rdf.py - convert an HPX log file into an RDF graph

\tusage: python rdf.py <log-filename>
"""

# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from hpx_log import HpxLog

from pyrple import Graph

import re
import sys

from templates import script_templates
from templates import search

def process_event(event, model):
  found = False
  for template in script_templates:
    if search(event, template):
      print template.in_english()
      model += template.as_rdf()
      found = True
      break

  if not found:
    pass #print "\tNo template for: %s" % (event['msg'])

def run(log_filename):
  log = HpxLog(log_filename)
  model = Graph()

  for event in log.get_events():
    process_event(event, model)

  rdf_file = open("run.rdf", "w")
  rdf_file.write(model.toRDFXML())
  rdf_file.close()

if __name__=="__main__":
  if (len(sys.argv) >= 2):
    log_filename = sys.argv[1]
    run(log_filename)
  else:
    print __doc__

