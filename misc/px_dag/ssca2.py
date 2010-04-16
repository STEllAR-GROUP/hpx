#!/usr/bin/python
"""ssca2.py - produce computational DAG for SSCA2 Benchmark application

\tusage: python ssca2.py <log-filename> 
"""

# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import re
import sys

from hpx_log import HpxLog

from px_dag import PxExecution
from px_dag import PxPhase
from px_dag import PxThread

import pxdm

g_action = '(future_value|got|set)'
g_gid = '{([^,]+), (.*)}.*'
re_future = re.compile('future_value::'+g_action+'\('+g_gid+'\)')

re_action = re.compile('\[(.*)\][^:]*(::\S*)')

def get_parent_tid(event):
  msg = event['msg']
  tid = msg[msg.index('(')+1:msg.index(')')]

  return tid

#def build_graph(thread, action):
#  # Kludge 1
#  action['--------'] = {'name':(0,'"not-a-thread"')}
#
#  tasks = [(t['name'],t['parent']) for t in thread.values()]
#
#  edges = []
#  vertices = []
#
#  for (u,v) in tasks:
#    cid = 'T'+u[1]
#    cid = cid.replace('-','_')
#    pid = 'T'+v[1]
#    pid = pid.replace('-','_')
#
#    label = '""'
#    if not pid in vertices:
#      if action.has_key(v[1]):
#        label = action[v[1]]['name'][1] 
#      vertices.append((pid,label))
#    if not cid in vertices:
#      if action.has_key(v[1]):
#        label = action[v[1]]['name'][1] 
#      vertices.append((cid,label))
#    edges.append((pid, cid))
#
#  return (edges, vertices)

def first_phase(phase):
  (thread, count) = phase.split('p')
  return 'p'.join([thread, '1'])

def next_phase(phase):
  (thread, count) = phase.split('p')
  return 'p'.join([thread, hex(int(count,16)+1)[2:]])

def previous_phase(phase):
  (thread, count) = phase.split('p')
  return 'p'.join([thread, hex(int(count,16)-1)[2:]])

def add_edge(edges, A, B):
  if not edges.has_key(A):
    edges[A] = [B]
  elif not B in edges[A]:
    edges[A].append(B)

def thread(t):
  return t.split('p')[0]

def node_name(phase):
  return "T%sp%s" % (phase.get_thread(), phase.get_id())

def parse_action_name(event, label, future, cadd):
  action = None
  m_future = re_future.search(event['msg'])
  if m_future:
    action = m_future.group(1)
    gid = m_future.group(3)

    if not future.has_key(gid):
      future[gid] = {action: cadd}
    else:
      future[gid][action] = cadd
  else:
    m_action = re_action.search(event['msg'])
    if m_action:
      action = m_action.group(1).lower() + m_action.group(2)
      if not 'eager_future' in action:
        if not label.has_key(thread(cadd)):
          label[thread(cadd)] = action

  return action

def build_model(app_run, log_filename):
  log = HpxLog(log_filename)
  
  threads = {}
  phases = {}

  label = {}
  future = {}

  for event in log.get_events():
    (child_loc, child_addr, child_gid) = event['thread']
    (parent_loc, parent_addr) = event['parent']

    ### Clean thread information
    if child_addr == '--------':
      child_addr = '0x0000000p0'
    if parent_addr == '--------':
      parent_addr = '0x0000000'

    child_addr = child_addr.replace('.','p')
    parent_addr = parent_addr

    action_name = parse_action_name(event, label, future, child_addr)

    # Create phase
    child_phase_id = 'T'+child_addr
    if not phases.has_key(child_phase_id):
      phases[child_phase_id] = PxPhase(child_phase_id)

    # Create threads
    child_thread_id = child_phase_id[:child_phase_id.find('p')]
    parent_thread_id = 'T'+parent_addr
    child_action_name = action_name 

    if not threads.has_key(child_thread_id):
      threads[child_thread_id] = PxThread(child_thread_id, child_action_name)
    elif not threads[child_thread_id].action_name() and action_name:
      threads[child_thread_id].set_action_name(action_name)
    if not threads.has_key(parent_thread_id):
      threads[parent_thread_id] = PxThread(parent_thread_id)

    # Add phase to thread
    threads[child_thread_id].add_phase(phases[child_phase_id])

  # Use futures information to set transitions and dependencies
  for f in future.values():
    A = B = C = D = E = False

    if f.has_key('got'):
      A = phases['T'+f['got']]
    if f.has_key('future_value'):
      C = phases['T'+f['future_value']]
    if f.has_key('set'):
      D = phases['T'+first_phase(f['set'])]
      E = phases['T'+f['set']]

    if A and B:
      pass #add_edge(edges, A, B)
    if C and D:
      # This says the phase that creates the future spawns
      # the first phase of the thread which contains the phase
      # that sets the future
      C.add_transition_to(D)
      D.add_dependency_on(C)
    if E and A:
      # This says the phase that sets the future is a dependency
      # for the phase that comes after the phase that gets the value
      E.add_transition_to(A)
      A.add_dependency_on(E)

  # Add threads to this application run
  app_run.add_threads(threads.values())

  return 

def cluster_vertices(edges):
  C = {}

  V = []
  for u in edges.keys():
    if not u in V:
      V.append(u)
    for v in edges[u]:
      if not v in V:
        V.append(v)

  for v in V:
    (t,p) = v.split('p')
    
    if not C.has_key(t):
      C[t] = [(v,p)]
    elif not (v,p) in C[t]:
      C[t].append((v,p))

  # Add thread phase connections
  for (t, T) in C.items():
    S = [(int(y,16),x) for (x,y) in T]
    S.sort()

    for i in range(len(S)-1):
      (p,t) = S[i]
      w = S[i+1][1]
      if not edges.has_key(t):
        #print "Adding %s -> %s" % (t, w)
        edges[t] = [w]
      elif not w in edges[t]:
        #print "Adding %s -> %s" % (t, w)
        edges[t].append(w)
    
  return C

def condense(edges, label):
  ignore_names = ['local_set', 'dist_set::','vertex::add_edge', 'vertex::vertex']
  merge_names = ['graph::add_edge()', 'graph::add_vertex()']

  # Do the ignoring first
  ignores = []
  for name in ignore_names:
    ignores += [x for x in label.keys() if name in label[x]]

  ignore_nodes = []
  for ignore in ignores:
    ignore_nodes += [n for n in edges.keys() if ignore in n]
 
  for n in ignore_nodes:
    del edges[n]
  
  for (k,v) in edges.items():
    edges[k] = [w for w in v if not w in ignore_nodes]

  # Now do the merging
  new_names = []
  for name in merge_names:
    phases = []
    merges = [x for x in label.keys() if name in label[x]]
    for merge_it in merges:
      phases += [n for n in edges.keys() if merge_it in n]

    new_name = 'T'+str(abs(hash(name)))+'p'+str(len(phases))
    new_names.append(new_name)
    edges[new_name] = []

    # Update edges
    for (k,v) in edges.items():
      # Set new in-edges
      new_v = [w for w in v if not w in phases]
      if len(new_v) < len(v):
        new_v.append(new_name)
        edges[k] = new_v
      if k in phases: # Set new out-edges
        edges[new_name] += edges[k]
        del edges[k]
    # Condense edges for new_name
    new_v = []
    for v in edges[new_name]:
      if not v in new_v:
        new_v.append(v)
    edges[new_name] = new_v          

    # Update labels
    for phase in phases:
      if phase in label.keys():
        del label[phase]
    label[new_name[:new_name.index('p')]] = name

  return new_names

def phase_number(phase):
  return phase.id()[phase.id().rfind('p')+1:]

def write_dot(app_run):
  cluster_number = 0

  print "digraph {"
  print "  rankdir=LR;"
  for thread in app_run.threads():
    cluster_number += 1
    print "  subgraph cluster%d {" % (cluster_number)
    print "    label = \"%s (%s)\";" % (thread.action_name(), thread.id())
    phases = thread.phases()
    phases.sort()
    for p in phases:
      print "    %s [label=\"%s\"];" % (p.id(), phase_number(p))
    if len(phases) > 0:
      print "    %s;" % (' -> '.join([p.id() for p in phases]))
    print "  }"
    for phase in thread.phases():
      for transition in phase.transitions():
        print "  %s -> %s;" % (phase.id(), transition.target().id())
  print "}"

def run(log_filename):
  app_run = PxExecution("SSCA2")

  build_model(app_run, log_filename)

  write_dot(app_run)

if __name__=="__main__":
  if (len(sys.argv) >= 2):
    log_file = sys.argv[1]
    run(log_file)
  else:
    print __doc__
