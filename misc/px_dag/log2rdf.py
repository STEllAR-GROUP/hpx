#!/usr/bin/python
"""log2rdf.py - convert an HPX log file into an RDF graph

Synopsis:
\tpython rdf.py <log-file> [-t <rdf-format>] [-o <rdf-file>] [-d]

Options:
\t-o\tWrite RDF output to <file> instead of stdout.

\t-t\tSet RDF output format to <rdf-format>. Options are 'rdfxml' for\n\t\tRDF/XML or 'ntriples'. Default is 'rdfxml'.

\t-d\tWrite extra information about lines that were not processed.
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

def search(event, template):
  match = template.re.search(event['msg'])
  if match:
    template.fill(event, match.groups())
    return True
  else:
    return False

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

