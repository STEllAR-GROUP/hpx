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

rdf = []

###

re_thread_id = re.compile('\(T([^/]+)/([^\)]+)/([^\)]+)\)')
re_parent_id = re.compile('P([^/]+)/([^\)]+)')
  
re_thread = re.compile('thread\(([^\)]+)\)')
re_desc = re.compile('description\(([^\)]*)\)')
re_desc_full = re.compile('description\(([^\)]*)::([^\)]*)\)')
re_state_old = re.compile('old state\(([^\)]*)\)')
re_state_new = re.compile('new state\(([^\)]*)\)')

###

def urlize(type, str):
  """Takes literal representation."""
  url = ''

  if type == 'action':
    url = str + 'Action'
  elif type == 'component':
    url = str + 'Component'
  elif type == 'fco':
    url = str + 'Fco'
  elif type == 'gid':
    url = str + 'Gid'
  elif type == 'locality':
    url = str + 'Locality'
  elif type == 'thread':
    url = str + 'Thread'
  else:
    print "ERROR: don't know how to urlize '%s'" % (type)
    exit()

  return url

def literalize(type, str):
  """Takes canonical representation."""

  literal = ''

  if 'action' in type:
    literal = ''.join([s.capitalize() for s in str.split('::')])
  elif type == 'component':
    literal = ''.join([s.capitalize() for s in str.split('::')])
  elif type == 'fco':
    literal = str
  elif type == 'gid':
    (msb, lsb) = str.split(':')
    literal = 'M'+msb+'L'+lsb
  elif type == 'locality':
    literal = str
  elif type == 'thread':
    literal = str
  else:
    print "ERROR: don't know how to literalize '%s'" % (type)
    exit()

  return literal

def canonicalize(type, str):
  """Takes raw representation."""

  if 'action' in type:
    res = str[:str.rfind('_action')]
  elif 'component' in type:
    parts = str.split('::')

    if 'server' in parts:
      parts.remove('server')
    if 'stubs' in parts:
      parts.remove('stubs')

    res = '::'.join(parts)
  elif 'fco' in type:
    if str == '--------':
      str = 'Null'
    res = str
  elif 'gid' in type:
    res = ':'.join(str)
  elif 'locality' in type:
    if str == '----':
      str = 'Null'
    res = 'L' + str
  elif 'thread' in type:
    (locale, thread) = str[:2]
    locale = canonicalize('locality', locale)
    if thread == '--------':
      thread = 'Null'
    res = locale + 'T' + thread
  else:
    print "ERROR: don't know how to canonicalize '%s'" % (type)

  return res

###

def parse_component(event, component):
  msg = event['msg'].split(':')

  name = msg[2][1:]
  type = msg[3][msg[3].index('[')+1:msg[3].index(']')]

  if not component.has_key(type):
    component[type] = {'name':name, 'type':type}

    # RDF
    name = canonicalize('component', name)
    url = urlize('component', literalize('component', name))
    rdf.append(':%s a px:Component; px:componentName "%s"; px:componentId "%s" .' % (url, name, type))
  else:
    print "Error: duplicate components!"
    exit(1)

def parse_object(event, object):
  re_mem_block = re.compile('component_memory_block\[([^\]]+)\]')
  re_type = re.compile('component\[([^\]]+)\]')
  
  re_size = re.compile('\(size: (\d+)\)')
  re_created = re.compile('created (\d+)')

  msg = event['msg']

  thread = event['thread']
  locale = thread[0]

  gid_str = msg[msg.index('{')+1:msg.index('}')]
  gid = tuple(gid_str.split(', '))

  # RDF
  o_name = urlize('gid', literalize('gid', canonicalize('gid', gid)))
  o_gid = canonicalize('gid', gid)
  rdf.append(':%s a px:Fco; px:gid "%s" .' % (o_name, o_gid))

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
    object[gid]['count'] = 1 #count
    object[gid]['thread'] = thread
    object[gid]['locale'] = locale
    
    # RDF
    locale = urlize('locality', literalize('locality', canonicalize('locality', locale)))
    rdf.append(':%s px:componentType [ px:componentId "%s" ] .' % (o_name, type))
    rdf.append(':%s px:locality :%s.' % (o_name, locale))
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

  locale = event['thread'][0]

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
    action[local_tid]['locale'] = locale
    action[local_tid]['name'] = name
    action[local_tid]['state'] = state

def parse_register_work(event):
  msg = event['msg']

  # Action
  # - name
  # - component

  m_desc_full = re_desc_full.search(msg)
  component = canonicalize('component', m_desc_full.group(1))
  action = canonicalize('action', m_desc_full.group(2))

  action_name = '::'.join((component, action))
  component_url = urlize('component', literalize('component', component))

  action_url = urlize('action', literalize('action', action_name))

  rdf.append(':%s a px:Action; px:actionName "%s"; px:component :%s .' % (action_url, action_name, component_url))

  # ActionEvent 
  # - action
  # - type
  # - parent thread
  # - source fco
  # [ a px:ActionEvent; px:action :XxxAction; px:actionType "yyy"; px:parent :ZzzThread ] .

  action_type = 'spawned'
  parent = urlize('thread', literalize('thread', canonicalize('thread', event['thread'])))
  source = urlize('fco', literalize('fco', canonicalize('fco', event['thread'][2])))
  rdf.append('[] a px:ActionEvent; px:action :%s; px:eventType "%s"; px:sourceThread :%s; px:sourceDatum [ px:localAddress "%s" ] .' % (action_url, action_type, parent, source))
  rdf.append(':%s a px:Thread; px:targetDatum [ px:localAddress "%s" ] .' % (parent, source))

def run(log_filename):
  log = HpxLog(log_filename)

  component = {}
  object = {}
  action = {}
  thread = {}
  locality = []

  rdf.append("@prefix px:  <http://px.cct.lsu.edu/pxo/0.1/> .")
  
  for event in log.get_events():
    child = event['thread']
    parent = event['parent']

    # RDF
    c_name = urlize('thread', literalize('thread', canonicalize('thread', child[:2])))
    p_name = urlize('thread', literalize('thread', canonicalize('thread', parent)))
    rdf.append(':%s a px:Thread; px:parent :%s .' % (c_name, p_name))
    rdf.append(':%s a px:Thread; px:child :%s .' % (p_name, c_name))

    # RDF
    (t_loc, t_gid, t_obj) = child
    t_gid = canonicalize('thread', (t_loc, t_gid))
    t_loc_id = canonicalize('locality', t_loc)
    t_loc = urlize('locality', literalize('locality', t_loc_id))
    t_obj = urlize('fco', literalize('fco', canonicalize('fco', t_obj)))
    rdf.append(':%s px:locality :%s; px:gid "%s"; px:source :%s .' % (c_name, t_loc, t_gid, t_obj))
    rdf.append(':%s a px:Locality; px:localityId "%s" .' % (t_loc, t_loc_id))

    # RDF
    (p_loc, p_gid) = parent
    rdf.append(':%s px:locality "%s"; px:gid "%s" .' % (p_name, p_loc, p_gid))

    if not thread.has_key(child):
      thread[child] = {}
      thread[child]['name'] = child[:2]
      thread[child]['parent'] = parent

    if 'dynamic loading succeeded' in event['msg']:
      parse_component(event, component)
    elif 'successfully created' in event['msg']:
      parse_object(event, object)
    elif '~thread' in event['msg']:
      parse_thread_dtor(event, action)
    elif 'TM' in event['module'] and 'register_work:' in event['msg']:
      parse_register_work(event)
    elif 'TM' in event['module'] and 'stop' in event['msg']:
      pass
    elif 'TM' in event['module'] and 'run:' in event['msg']:
      pass
    elif 'TM' in event['module'] and 'add_new:' in event['msg']:
      pass
    elif 'about to stop services' in event['msg']:
      pass
    elif 'exiting wait' in event['msg']:
      pass
    elif 'queues empty' in event['msg']:
      pass
    elif 'connection_cache' in event['msg']:
      pass
    elif 'HPX threads' in event['msg']:
      pass
    elif 'stopping timer pool' in event['msg']:
      pass
    elif 'tfunc' in event['msg']:
      parse_tfunc(event, action)
    else:
      pass #print event

  for thr in thread.values():
    locale = thr['name'][0]
    if not locale in locality:
      locality.append(locale)

  # Write out RDF
  rdf_out = open(log_filename+'.n3', 'w')
  rdf_out.writelines([line+'\n' for line in rdf])

  return (component, object, action, thread, locality)

if __name__=="__main__":
  if len(sys.argv) == 2:
    filename = sys.argv[1]
    run(filename)
  else:
    print __doc__
