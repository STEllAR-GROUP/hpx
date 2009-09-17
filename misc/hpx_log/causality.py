#!/usr/bin/python
"""causality.py - produce graph representing threads and interrelationships.

\tusage: python causality.py <log-filename> [<search-string>]
"""

# Copyright (c) 2009-2010 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import sys

from hpx_log import HpxLog

def get_parent_tid(event):
  msg = event['msg']
  tid = msg[msg.index('(')+1:msg.index(')')]

  return tid

def run(filename, exp):
  log = HpxLog(filename)

  vertices = []
  edges = []

  for event in log.get_events():
    if event.has_key('msg') and exp in event['msg']:
      source = '/'.join(event['parent'])
      target = '/'.join(event['thread'])
      src = 0
      tgt = 0

      if not source in vertices:
        vertices.append(source)
        src = len(vertices)-1
      else:
        src = vertices.index(source)
      if not target in vertices:
        vertices.append(target)
        tgt = len(vertices)-1
      else:
        tgt = vertices.index(target)

      if (src,tgt) not in edges:
        edges.append((src,tgt))

  # Write DOT to to file
  dot_filename = filename + '.dot'
  dot = open(dot_filename, 'w')

  dot.write("digraph {\n")
 
  #for i in range(len(vertices)):
  #  dot.write("  %s [shape=point];\n" % (i))

  for (u,v) in edges:
    u_name = 'T' + vertices[u].replace('/','_')
    u_name = u_name.replace('-','_')
    v_name = 'T' + vertices[v].replace('/','_')
    v_name = v_name.replace('-','_')
    dot.write("  %s -> %s;\n" % (u_name, v_name))

  dot.write("}\n")

  # Write out some additional information
  print "Nodes:"
  for v in vertices:
    print "  %d\t%s" % (vertices.index(v), v)

  print "Arcs:"
  for (u,v) in edges:
    print "  %d\t(%s, %s)" % (edges.index((u,v)), vertices[u], vertices[v])

if __name__=="__main__":
  if (len(sys.argv) == 3):
    exp = sys.argv[2]
  else:
    exp = ''

  if (len(sys.argv) >= 2):
    log_file = sys.argv[1]
    run(log_file, exp)
  else:
    print __doc__
