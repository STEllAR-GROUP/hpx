#!/usr/bin/python
"""pxdm.py -
"""

# Copyright (c) 2009-2010 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import re
import sys

from hpx_log import HpxLog

###

re_thread_id = re.compile('\([TP]([^/]+)/([^\)]+)\)')
  
re_thread = re.compile('thread\(([^\)]+)\)')
re_desc = re.compile('description\(([^\)]*)\)')
re_desc_full = re.compile('description\(([^\)]*)::([^\)]*)\)')
re_state_old = re.compile('old state\(([^\)]*)\)')
re_state_new = re.compile('new state\(([^\)]*)\)')

###

def parse_component(event, component):
  msg = event['msg'].split(':')

  name = msg[2][1:]
  type = msg[3][msg[3].index('[')+1:msg[3].index(']')]

  if not component.has_key(type):
    component[type] = {'name':name, 'type':type}
  else:
    print "Error: duplicate components!"
    exit(1)

def parse_object(event, object):
  re_mem_block = re.compile('component_memory_block\[([^\]]+)\]')
  re_type = re.compile('component\[([^\]]+)\]')
  
  re_size = re.compile('\(size: (\d+)\)')
  re_created = re.compile('created (\d+)')

  msg = event['msg']

  gid_str = msg[msg.index('{')+1:msg.index('}')]
  gid = tuple(gid_str.split(', '))

  type = 0
  count = 0

  m_mem_block = re_mem_block.search(msg)
  if m_mem_block:
    type = m_mem_block.group(1)
    m_size = re_size.search(msg)
    count = int(m_size.group(1))
  else:
    m_type = re_type.search(msg)
    type = m_type.group(1)

    m_created = re_created.search(msg)
    count = int(m_created.group(1))

  if not object.has_key(gid):
    object[gid] = {}
    object[gid]['gid'] = gid
    object[gid]['type'] = type
    object[gid]['count'] = count
  else:
    print "Error: duplicate objects"
    exit(1)

def parse_thread_dtor(event, action):
  msg = event['msg']

  name = ''

  m_thread = re_thread.search(msg)
  local_tid = m_thread.group(1)

  m_desc = re_desc_full.search(msg)
  if m_desc:
    name = m_desc.groups()
  else:
    m_desc = re_desc.search(msg)
    name = ('',m_desc.group(1))

  # Not actually storing this information
  pass

def parse_tfunc(event, action):
  msg = event['msg']

  if 'tfunc(0): start' in msg:
    return

  m_thread = re_thread.search(msg)
  local_tid = m_thread.group(1)

  name = ()
  m_desc_full = re_desc_full.search(msg)
  if m_desc_full:
    name = m_desc_full.groups()
  else:
    m_desc = re_desc.search(msg)
    name = ('', m_desc.group(1))

  state = ''
  
  m_state = re_state_new.search(msg)
  if m_state:
    state = m_state.group(1)
  else:
    state = 'activated'

  if not action.has_key(local_tid):
    action[local_tid] = {}
    action[local_tid]['thread'] = local_tid
    action[local_tid]['name'] = name
    action[local_tid]['state'] = state

def run(log_filename):
  log = HpxLog(log_filename)

  component = {}
  object = {}
  action = {}
  thread = {}

  for event in log.get_events():
    child = event['thread']
    parent = event['parent']
    if not thread.has_key(child):
      thread[child] = {}
      thread[child]['name'] = child
      thread[child]['parent'] = parent

    if 'dynamic loading succeeded' in event['msg']:
      parse_component(event, component)
    elif 'successfully created' in event['msg']:
      parse_object(event, object)
    elif '~thread' in event['msg']:
      parse_thread_dtor(event, action)
    elif 'TM' in event['level'] and 'stop' in event['msg']:
      pass
    elif 'about to stop services' in event['msg']:
      pass
    elif 'exiting wait' in event['msg']:
      pass
    elif 'queues empty' in event['msg']:
      pass
    elif 'connection_cache' in event['level']:
      pass
    elif 'HPX threads' in event['msg']:
      pass
    elif 'stopping timer pool' in event['msg']:
      pass
    elif 'tfunc' in event['msg']:
      parse_tfunc(event, action)
    else:
      pass #print event

  return (component, object, action, thread)

if __name__=="__main__":
  if len(sys.argv) == 2:
    filename = sys.argv[1]
    run(filename)
  else:
    print __doc__
