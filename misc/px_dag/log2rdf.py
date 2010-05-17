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

def process_event(event, model, show_english, show_missing):
  found = False
  for template in script_templates:
    if search(event, template):
      model += template.as_rdf()
      if show_english:
        eng = template.in_english()
        if eng:
          sys.stderr.write(eng + "\n")
      found = True
      break

  if not found and show_missing:
    sys.stderr.write("\tNo template for: %s\n" % (event['msg']))

def run(options, log_filename):
  log = HpxLog(log_filename)
  model = Graph()

  for event in log.get_events():
    process_event(event, model, options.show_english, options.show_missing)

  out = sys.stdout
  if options.outfile:
    out = open(options.outfile, 'w')

  if options.outformat == 'rdfxml':
    out.write(model.toRDFXML())
  elif options.outformat == 'ntriples':
    out.write(model.serialize())
  else:
    sys.stderr.write("Unknown output format, defaulting to ntriples.\n")
    out.write(model.serialize())

def setup_options():
  usage = "usage: %prog [options] logfile"
  parser = OptionParser(usage=usage)
  parser.add_option("-o", "--outfile", dest="outfile",
                    help="write RDF output to FILE", metavar="FILE")
  parser.add_option("-f", "--outformat", dest="outformat",
                    default="ntriples",
                    help="RDF output format: 'ntriples' [default] or 'rdfxml'")
  parser.add_option("-e", "--english", action="store_true",
                    dest="show_english", default=False,
                    help="Show matched log events in 'plain English' (written to stderr)")
  parser.add_option("-m", "--missing", action="store_true", 
                    dest="show_missing", default=False,
                    help="Show unmatched log events (written to stderr)")

  return parser

if __name__=="__main__":
  parser = setup_options()
  (options, args) = parser.parse_args()

  if (len(args) == 1):
    log_filename = args[0]
    run(options, log_filename)
  else:
    parser.print_help()

