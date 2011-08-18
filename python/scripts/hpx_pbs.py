#!/usr/bin/env python
#
# Copyright (c) 2011 Bryce Lelbach
#
# Based on code by Robey Pointer
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from __future__ import with_statement

from sys import exit, argv, version_info, executable

from os import getpid, environ
from os.path import join, dirname, abspath, expanduser

from re import compile

from optparse import OptionParser

from socket import socket, gethostbyname, error 
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

# {{{ version check
if version_info[0] < 2:
  report("Python %d.%d is too old, Python 2.5 or newer is required."
        % (version_info[:2]))
  exit(1) 
elif 2 == version_info[0] and version_info[1] < 5:
  report("Python %d.%d is too old, Python 2.5 or newer is required."
        % (version_info[:2]))
  exit(1) 
# }}}

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

def prepare_args(cmd):
  s = ''
  for e in cmd:
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
      report("Connect failed with %s (\"%s\")." % (self.hostname, str(e)))
      return None

    self.transport = Transport(sock)

    try:
      self.transport.start_client()
    except SSHException, e:
      report( "SSH negotiation failed with %s (\"%s\")."
            % (self.hostname, str(e)))
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

def execute_function(user, hostname, port, keys, path, cmd, callback, timeout):
  with transport_raii(hostname, port) as transport:
    if None == transport:
      return  
  
    key = transport.get_remote_server_key()
    if not keys.has_key(hostname):
      report("Unknown host key for %s." % hostname)
    elif not keys[hostname].has_key(key.get_name()):
      report("Unknown host key for %s." % hostname)
    elif keys[hostname][key.get_name()] != key:
      report("Host key has changed.")
      return 
  
    agent_auth(transport, user)
  
    if not transport.is_authenticated():
      key_auth(transport, user, path)
  
    if not transport.is_authenticated():
      report("Authentication failed.")
      return 

    try:  
      with channel_raii(transport) as channel:  
        channel.set_combine_stderr(True)
        # We add 30 seconds to the timeout to give hpx_invoke time to kill the
        # process on the remote end.
        channel.settimeout(float(timeout) + 30.0)
        channel.exec_command(cmd)
  
        output = ''
  
        while True:
          output_buffer = channel.recv(256)
  
          if not output_buffer:
            break
  
          output += output_buffer
  
        r = channel.recv_exit_status()
  
        callback(r, output)

    except error:
      callback(1, 'Socket timeout')

def remote_execute(user, hostname, port, keys, key, cmd, callback, timeout):
  worker = Thread(target=execute_function
         , args=(user, hostname, port, keys, key, cmd, callback, timeout))
  worker.start()
  return worker

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

def load_node_file(file, default_port=0):
  nodes = {}

  try:
    with open(file) as f:
      for line in f:
        line = strip(line, whitespace).lower()

        # ignore empty lines
        if 0 == len(line):
          pass
        else:
          line = parse_ip(line, default_port)
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

def rstrip_last(s, chars):
  if s[-1] in chars:
    return s[:-1]
  else:
    return s

class io_callback:
# {{{
  mutex = None
  node = None

  def __init__(self, mutex, node):
    self.mutex = mutex
    self.node = deepcopy(node)

  def __call__(self, r, output):
    with scoped_lock(mutex) as lock:
      print "########################################"
      print "node   == %s:%d" % self.node
      print "return == %d"    % r

      if output:
        print "#OUTPUT#################################"
        print rstrip_last(output, '\n')

      print "########################################\n"
# }}}

# {{{ main

# {{{ default HPX location discovery
# use the default HPX install directory as our fallback
location = '/opt/hpx'

try:
  from sys import getwindowsversion
  location = 'C:/Program Files/hpx' 
except:
  pass

# if we were invoked with a fully qualified path, default to the prefix of
# that path
try:
  # argv[0] is PREFIX/bin/hpx_pbs or PREFIX/bin/hpx_pbs.py if we've been invoked
  # with a fully qualified path
  location = dirname(dirname(abspath(argv[0])))
except:
  pass
# }}}

# {{{ command line handling
usage = "Usage: %prog [options] program [program-arguments]" 

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

parser.add_option("--hpx-location",
                  action="store", type="string", dest="hpx_location",
                  help="HPX installation prefix usuable on all nodes "+
                       "(default: $HPX_LOCATION or %s)" % location)

parser.add_option("--hpx-console-options",
                  action="store", type="string", dest="hpx_console_options",
                  help="Options specific to the console locality")

parser.add_option("--hpx-agas-options",
                  action="store", type="string", dest="hpx_agas_options",
                  help="Options specific to the HPX AGAS locality")

parser.add_option("--no-hpx-location",
                  action="store_true", dest="no_hpx_location", default=False,
                  help="Don't add a prefix to the program invocation")

parser.add_option("--no-hpx-invoke",
                  action="store_true", dest="no_hpx_invoke", default=False,
                  help="Don't use hpx_invoke")

parser.add_option("--nodes",
                  action="store", type="string",
                  dest="nodes", help="PBS nodefile (default: $PBS_NODEFILE)")

parser.add_option("--timeout",
                  action="store", type="int",
                  dest="timeout", default=3600,
                  help="Program timeout (seconds)")

parser.add_option("--dry-run",
                  action="store_true", dest="dry_run", default=False,
                  help="Print out commands, don't run them")

(options, cmd) = parser.parse_args()

if 0 == len(cmd):
  report("No command specified.")
  exit(1) 
# }}}

# {{{ HPX location
if options.no_hpx_location:
  if not None == options.hpx_location:
    report("--no-hpx-invoke and --hpx-location are incompatible")
    exit(1)
elif not None == options.hpx_location: 
  location = expanduser(options.hpx_location)
elif environ.has_key('HPX_LOCATION'):
  location = expanduser(environ['HPX_LOCATION'])
# }}}

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
  nodes = load_node_file(expanduser(options.nodes), options.hpx_port)
elif environ.has_key('PBS_NODEFILE'):
  nodes = load_node_file(expanduser(environ['PBS_NODEFILE']), options.hpx_port)

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
agas = nodes.items()[0][0]

if not None == options.agas:
  agas = parse_ip(options.agas, options.hpx_port)

  if not nodes.has_key(agas):
    report("AGAS locality %s:%d is not available for this job." % agas)
    exit(1)
# }}}

# {{{ console
console = nodes.items()[0][0]

if not None == options.console:
  console = parse_ip(options.console, options.hpx_port)

  if not nodes.has_key(console):
    report("Console locality %s:%d is not available for this job." % console)
    exit(1)
# }}}

# {{ prepare command 
bin = cmd[0]
args = prepare_args(cmd[1:])

invoc = ''
bindir = ''

if not options.no_hpx_location:
  bindir = join(location, 'bin')

if not options.no_hpx_invoke:
  invoc = executable + ' ' + join(bindir, 'hpx_invoke.py')
  invoc += ' --timeout=%d' % options.timeout
  invoc += ' --program=\'' + join(bindir, bin)

else:
  invoc = join(bindir, bin)

if len(args):
  invoc += ' ' + args 
# }}}

# {{{ scheduler loop
cmds = []

for node in nodes.iterkeys():
  local_cmd = invoc  
  local_cmd += ' -t%d'                  % nodes[node]
  local_cmd += ' -Ihpx.agas.address=%s' % gethostbyname(agas[0])
  local_cmd += ' -Ihpx.agas.port=%d'    % agas[1]
  local_cmd += ' -l%d'                  % len(nodes)

  if agas == node:
    if options.hpx_agas_options:
      local_cmd += ' ' + options.hpx_agas_options
    local_cmd += ' -r'
    local_cmd += ' -x%s:%d' % (gethostbyname(agas[0]), agas[1]) 
  elif console[0] == node:
    local_cmd += ' -x%s:%d' % (gethostbyname(console[0]), console[1]) 
  else:
    local_cmd += ' -x%s:%d' % (gethostbyname(node[0]), node[1]) 

  if not console == node:
    local_cmd += ' -w' 
  else:
    if options.hpx_console_options:
      local_cmd += ' ' + options.hpx_console_options
    local_cmd += ' -c'

  if not options.no_hpx_invoke:
    local_cmd += '\''

  if options.dry_run:
    print '%s:%d == %s' % (node[0], node[1], local_cmd)
  else:
    cb = io_callback(mutex, node)
    timeout = options.timeout
    cmds.append((user, node[0], port, keys, key, local_cmd, cb, timeout))
    print '%s:%d == %s' % (node[0], node[1], local_cmd)
# }}}

if not options.dry_run:
  print

  for cmd in cmds:
    threads.append(remote_execute(*cmd))

  for thread in threads:
    thread.join()
# }}}

