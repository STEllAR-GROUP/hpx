#!/usr/bin/python
"""dependency.py - 

\tusage: python dependency.py <log-filename>
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
  action['--------'] = {'name':'not-a-thread'}

  tasks = [(t['name'],t['parent']) for t in thread.values()]

  edges = []
  vertices = []

  for (u,v) in tasks:
    cid = 'T'+u[1]
    cid = cid.replace('-','_')
    pid = 'T'+v[1]
    pid = pid.replace('-','_')

    p_label = 'x'
    c_label = 'x'
    if not pid in vertices:
      if action.has_key(v[1]):
        p_label = action[v[1]]['name'][1] 
      vertices.append((p_label,p_label))
    if not cid in vertices:
      if action.has_key(u[1]):
        c_label = action[u[1]]['name'][1] 
      vertices.append((c_label,c_label))
    
    e = (p_label, c_label)
    if not e in edges:
      edges.append(e)

  return (edges, vertices)

def run(log_filename):
  (component, object, action, thread) = pxdm.run(log_filename)

  (edges, vertices) = build_graph(thread, action)

  print "digraph {"
  for (u,w) in vertices:
    print "  %s [label=%s];" % (u,w)
  for (u,v) in edges:
    print "  %s -> %s;" % (u,v)
  print "}"

if __name__=="__main__":
  if (len(sys.argv) == 2):
    log_file = sys.argv[1]

    run(log_file)
  else:
    print __doc__
