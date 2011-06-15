#!/usr/bin/env python
#
#  Copyright (c) 2011 Bryce Lelbach
#
#  Based on code by Robey Pointer
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from __future__ import with_statement

from sys import exit

from os import getpid, environ
from os.path import join, expanduser

from re import compile

from optparse import OptionParser

from socket import socket
from socket import AF_INET, SOCK_STREAM 

from paramiko import Agent, RSAKey, DSSKey, Transport
from paramiko import SSHException, PasswordRequiredException
from paramiko.util import log_to_file, load_host_keys

from getpass import getuser

from string import strip, whitespace, letters, digits

from types import StringType
    
from threading import Thread, RLock

from copy import deepcopy

def report(str):
  print str

def agent_auth(transport, username):
  agent = Agent()
  agent_keys = agent.get_keys()

  if len(agent_keys) == 0:
    report("Couldn't contact SSH agent for %s." % username)
    return False

  for key in agent_keys:
    try:
      transport.auth_publickey(username, key)
      return True
    except SSHException:
      pass
  
  report("Couldn't contact SSH agent for %s." % username)
  return False

def key_auth(transport, username, path):
  key = None

  try:
    key = RSAKey.from_private_key_file(path)

  except PasswordRequiredException:
    report("RSA key %s requires a password." % path)
    return False

  except:
    try:
      key = DSSKey.from_private_key_file(path)

    except PasswordRequiredException:
      report("DSS key %s requires a password." % path)
      return False

    except:
      report("%s is not an SSH key." % path)
      return False

  transport.auth_publickey(username, key)
  return True

def prepare_command(cmd, quote = '"'):
  exclude = letters + digits + '-+='
  s = ''
  for e in cmd:
    if type(e) is not StringType:
      e = str(e)
    for c in e:
      if c not in exclude:
        s += ' ' + quote + e + quote
        break
      else:
        s += ' ' + e
  return s

class transport_raii:
  transport = None
  hostname = None
  port = None

  def __init__(self, hostname, port):
    self.hostname = hostname
    self.port = port

  def __enter__(self):
    try:
      sock = socket(AF_INET, SOCK_STREAM)
      sock.connect((self.hostname, self.port))
    except Exception, e:
      report("Connect failed (\"%s\")." % str(e))
      return None

    self.transport = Transport(sock)

    try:
      self.transport.start_client()
    except SSHException:
      print("SSH negotiation failed.")
      return None

    return self.transport

  def __exit__(self, type, value, traceback): 
    if not None == self.transport:
      self.transport.close()

class channel_raii:
  channel = None
  transport = None

  def __init__(self, transport):
    self.transport = transport

  def __enter__(self):
    try:
      c = self.transport.open_session()
    except Exception, e:
      report("Couldn't open an SSH session (\"%s\")." % str(e))
      return None

    self.channel = c
    return self.channel

  def __exit__(self, type, value, traceback):
    if not None == self.channel:
      self.channel.close()

def execute_function(username, hostname, port, key, cmd, callback):
  with transport_raii(hostname, port) as transport:
    if None == transport:
      return  
  
    key = transport.get_remote_server_key()
    if not keys.has_key(hostname):
      report("Unknown host key.")
    elif not keys[hostname].has_key(key.get_name()):
      report("Unknown host key.")
    elif keys[hostname][key.get_name()] != key:
      report("Host key has changed.")
      return 
  
    agent_auth(transport, username)
  
    if not transport.is_authenticated():
      key_auth(transport, username, key)
  
    if not transport.is_authenticated():
      report("Authentication failed.")
      return 
  
    with channel_raii(transport) as channel:  
      channel.exec_command(cmd)

      stdout = ''
      stderr = ''

      while True:
        stdout_buffer = channel.recv(256)
        stderr_buffer = channel.recv_stderr(256)

        if not stdout_buffer and not stderr_buffer:
          break

        stdout += stdout_buffer
        stderr += stderr_buffer

      callback(stdout, stderr)

def remote_execute(username, hostname, port, key, cmd, callback):
  worker = Thread(target=execute_function
                , args=(username, hostname, port, key, cmd, callback))
  worker.start()
  return worker

def load_node_file(file):
  nodes = {}

  try:
    with open(file) as f:
      for line in f:
        line = strip(line, whitespace)
        if nodes.has_key(line):
          nodes[line] = nodes[line] + 1
        else:
          nodes[line] = 1

  except IOError:
    report("Unable to open PBS node file %s." % file)

  return nodes

class scoped_lock:
  lock = None

  def __init__(self, lock):
    self.lock = lock

  def __enter__(self):
    self.lock.acquire()
    return self.lock

  def __exit__(self, type, value, traceback):
    if not None == self.lock:
      self.lock.release()

class io_callback:
# {{{
  mutex = None
  cmd = None
  node = None

  def __init__(self, mutex, cmd, node):
    self.mutex = mutex
    self.cmd = cmd
    self.node = deepcopy(node)

  def __call__(self, stdout, stderr):
    with scoped_lock(mutex) as lock:
      print "Executing \"%s\" on %s:" % (self.cmd, self.node)

      print "=stdout======================================================="
      if stdout:
        print stdout

      print "=stderr======================================================="
      if stderr:
        print stderr

      print "=============================================================="
# }}}

def parse_ip(ip, default_port=0):
# {{{
  regex = compile(r'^(\[[a-fA-F0-9:]+\]|[a-zA-Z0-9\-.]+):?([0-9]*)').match(ip)
  
  if (regex):
    port = regex.expand(r'\2')

    try:
      port = int(port)
    except:
      port = default_port

    host = regex.expand(r'\1')

    if '[' == host[0] and ']' == host[-1]:
      host = host[1:-1]

    return (host, port)
  else:
    return None 
# }}}

# {{{ main
usage = "usage: %prog [options] command" 

parser = OptionParser(usage=usage)

parser.add_option("--ssh-user",
                  action="store", type="string", dest="ssh_user",
                  default=getuser(), help="Username")

parser.add_option("--ssh-port",
                  action="store", type="int", dest="ssh_port",
                  default=22, help="SSH port")

parser.add_option("--ssh-key",
                  action="store", type="string", dest="ssh_key",
                  default="~/.ssh/id_rsa", 
                  help="SSH key")

parser.add_option("--ssh-known-hosts",
                  action="store", type="string", dest="known_hosts",
                  default="~/.ssh/known_hosts", 
                  help="SSH known hosts file")

parser.add_option("--hpx-agas",
                  action="store", type="string",
                  dest="agas", help="Address of the HPX AGAS locality")

parser.add_option("--hpx-console",
                  action="store", type="string",
                  dest="console", help="Address of the HPX console locality")

parser.add_option("--hpx-port",
                  action="store", type="int", dest="hpx_port",
                  default=7910, help="Default HPX IP port")

parser.add_option("--nodes",
                  action="store", type="string",
                  dest="nodes", help="PBS nodefile")

parser.add_option("--verbatim",
                  action="store_true",
                  dest="verbatim", default=False,
                  help="Don't quote command elements")

parser.add_option("--dry-run",
                  action="store_true",
                  dest="dry_run", default=False,
                  help="Print out commands, don't run them")

(options, cmd) = parser.parse_args()

if 0 == len(cmd):
  report("No command specified.")
  exit(1) 

# {{{ known hosts
known_hosts = expanduser(options.known_hosts)
  
keys = {}
  
try:
  keys = load_host_keys(known_hosts)
except IOError:
  report("Unable to open SSH known hosts file %s." % known_hosts)
# }}}

# {{{ node file
nodes = {}

if not None == options.nodes: 
  nodes = load_node_file(expanduser(options.nodes))
elif environ.has_key('PBS_NODEFILE'):
  nodes = load_node_file(expanduser(environ['PBS_NODEFILE']))

if 0 == len(nodes):
  report("No nodes specified.")
  exit(1)
# }}}

threads = []
mutex = RLock() 

# {{{ ssh
port = 22

if not None == options.ssh_port:
  port = options.ssh_port

user = options.ssh_user

key = expanduser(options.ssh_key)
# }}}

# {{{ agas
agas = (nodes.items()[0][0], options.hpx_port)

if not None == options.agas:
  agas = parse_ip(options.agas, options.hpx_port)

  if not nodes.has_key(agas[0]):
    report("AGAS locality %s is not available for this job." % agas[0])
    exit(1)
# }}}

# {{{ console
console = (nodes.items()[0][0], options.hpx_port)

if not None == options.console:
  console = parse_ip(options.console, options.hpx_port)

  if not nodes.has_key(console[0]):
    report("Console locality %s is not available for this job." % console[0])
    exit(1)
# }}}

# {{ quote command line
if options.verbatim:
  cmd = cmd[0] + prepare_command(cmd[1:], '')
else:
  cmd = cmd[0] + prepare_command(cmd[1:])
# }}}

# {{{ scheduler loop
for node in nodes.iterkeys():
  local_cmd = cmd  
  local_cmd += ' "-t%d"'                  % nodes[node]
  local_cmd += ' "-Ihpx.agas.address=%s"' % agas[0]
  local_cmd += ' "-Ihpx.agas.port=%d"'    % agas[1]
  local_cmd += ' "-l%d"'                  % len(nodes)

  if agas[0] == node:
    local_cmd += ' "-r"'
    local_cmd += ' "-x%s:%d"' % (agas[0], agas[1]) 
  elif console[0] == node:
    local_cmd += ' "-x%s:%d"' % (console[0], console[1]) 
  else:
    local_cmd += ' "-x%s:%d"' % (node, options.hpx_port) 

  if not console[0] == node:
    local_cmd += ' "-w"' 

  if options.dry_run:
    print '%s: %s' % (node, local_cmd)
  else:
    cb = io_callback(mutex, local_cmd, node)
    threads.append(remote_execute(user, node, port, key, local_cmd, cb))
# }}}

if options.dry_run:
  for thread in threads:
    thread.join()
# }}}

