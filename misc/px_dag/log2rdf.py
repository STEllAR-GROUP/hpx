#!/usr/bin/python
"""log2rdf.py - convert an HPX log file into an RDF graph
"""

# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from hpx_log import HpxLog

from optparse import OptionParser

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

def setup_options():
  usage = "usage: %prog [options] logfile"
  parser = OptionParser(usage=usage)
  parser.add_option("-o", "--outfile", dest="outfile",
                    help="write RDF output to FILE", metavar="FILE")
  parser.add_option("-f", "--outformat", dest="outformat",
                    default="rdfxml",
                    help="RDF output format: 'ntriples' or 'rdfxml' [default]")
  parser.add_option("-m", "--missing", action="store_true", 
                    dest="show_missing", default="false",
                    help="Show unmatched log events.")

  return parser

if __name__=="__main__":
  parser = setup_options()
  (options, args) = parser.parse_args()

  if (len(args) == 1):
    log_filename = args[0]
    run(log_filename)
  else:
    parser.print_help()

