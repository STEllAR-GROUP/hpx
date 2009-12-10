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

import pxdm

def get_parent_tid(event):
  msg = event['msg']
  tid = msg[msg.index('(')+1:msg.index(')')]

  return tid

def build_graph(thread, action):
  # Kludge 1
  action['--------'] = {'name':(0,'"not-a-thread"')}

  tasks = [(t['name'],t['parent']) for t in thread.values()]

  edges = []
  vertices = []

  for (u,v) in tasks:
    cid = 'T'+u[1]
    cid = cid.replace('-','_')
    pid = 'T'+v[1]
    pid = pid.replace('-','_')

    label = '""'
    if not pid in vertices:
      if action.has_key(v[1]):
        label = action[v[1]]['name'][1] 
      vertices.append((pid,label))
    if not cid in vertices:
      if action.has_key(v[1]):
        label = action[v[1]]['name'][1] 
      vertices.append((cid,label))
    edges.append((pid, cid))

  return (edges, vertices)

def run(log_filename, node_style, arc_style):
  (component, object, action, thread, locality) = pxdm.run(log_filename)

  (edges, vertices) = build_graph(thread, action)

  print "digraph {"
  for (u,w) in vertices:
    print "  %%s %s;" % (node_style) % (u,w)
  for (u,v) in edges:
    print "  %s -> %s;" % (u,v)
  print "}"

if __name__=="__main__":
  node_style = '[label=%s]'
  arc_style = ''

  if (len(sys.argv) >= 2):
    log_file = sys.argv[1]

    if "--point" in sys.argv:
      node_style = '[label=%s]' # shape=point,

    run(log_file, node_style, arc_style)
  else:
    print __doc__
