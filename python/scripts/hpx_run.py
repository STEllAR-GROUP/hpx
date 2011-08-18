#!/usr/bin/env python
"""hpx_run.py - spawn multi-locality HPX jobs
"""

#  Copyright (c) 2010 Dylan Stark
# 
#  Distributed under the Boost Software License, Version 1.0. (See accompanying 
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from copy import copy

import os
import sys

from socket import gethostbyname

import subprocess

from optparse import OptionParser

class LocalitySet(dict):
  """
  A colleciton of localities.
  """
  def __init__(self, *args):
    dict.__init__(self, args)

  def __getitem__(self, key):
    if not dict.__contains__(self, key):
      dict.__setitem__(self, key, Locality(key))
    return dict.__getitem__(self, key)

  def __iter__(self):
    return dict.__iter__(self)

  def __str__(self):
    str = "Locality set:\n"
    str += "\t%d localities:\n" % (dict.__len__(self))
    for (locality_id,locality) in dict.items(self):
      str += "\tLocality '%s' with %d threads\n" % (
          locality_id, locality.num_threads)

    return str

  def load_from_system_view(self, system):
    for node_name in system:
      locality_id = "L%s" % (node_name)

      self[locality_id].num_threads = system[node_name].num_cores

  def load_from_locality_spec(self, locality_spec):
    (num_localities, num_threads) = parse_localities_spec(locality_spec)

    for locality in range(num_localities):
      locality_id = "L%d" % (locality)

      self[locality_id].num_threads = num_threads
 
class Locality:
  """
  A locality.
  """
  def __init__(self, id):
    self.id = id
    self.ip = None
    self.num_threads = 0
    self.schedule_policy = None

  def __str__(self):
    str = "L%s - %s OS threads, with %s scheduling" % (self.id, self.num_threads, self.schedule_policy)

    return str

class System(dict):
  """
  A collection of resources (e.g., nodes).
  """
  def __init__(self, *args):
    dict.__init__(self, args)

  def __getitem__(self, key):
    if not dict.__contains__(self, key):
      dict.__setitem__(self, key, Node(key))
    return dict.__getitem__(self, key)

  def __iter__(self):
    return dict.__iter__(self)

  def __str__(self):
    str = "System view:\n"
    str += "\t%d nodes:\n" % (dict.__len__(self))
    for (node_name,node) in dict.items(self):
      str += "\tNode '%s' with %d cores\n" % (node_name, node.num_cores)

    return str

  def load_from_machine_file(self, machine_filename):
    self.is_virtual = False

    machine_file = open(machine_filename, 'r')
    for line in machine_file:
      node_name = line.strip()
      self[node_name].num_cores += 1
     
    for node_name in self:
      ip = gethostbyname(node_name)
      if len(ip) > 0:
        self[node_name].ip = ip
      else:
        self[node_name].ip = '127.0.0.1'
        self.is_virtual = True
  
  def load_from_localities_spec(self, localities_spec):
    self.is_virtual = True

    (num_localities, num_threads) = parse_localities_spec(localities_spec)

    for locality in range(num_localities):
      node_name = "node%d" % (locality)

      self[node_name].is_virtual=True
      self[node_name].num_cores = num_threads
      self[node_name].ip = '127.0.0.1'

class Node:
  """
  A compute node.
  """
  def __init__(self, name, is_virtual=False):
    self.name = name
    self.num_cores = 0
    self.is_virtual=is_virtual

  def __str__(self):
    str = "%s - %d cores @ %s" % (self.name, self.num_cores, self.ip)
    return str

class DistributedRuntime(dict):
  """
  A collection of local runtime instances.
  """
  def __init__(self, *args):
    dict.__init__(self, args)

  def __getitem__(self, key):
    if not dict.__contains__(self, key):
      dict.__setitem__(self, key, LocalRuntime(key))
    return dict.__getitem__(self, key)

  def __iter__(self):
    return dict.__iter__(self)

  def __str__(self):
    str = "Distributed runtime:\n"
    str += "\t%d local instances:\n" % (dict.__len__(self))
    for (runtime_id,runtime) in dict.items(self):
      str += "\tRuntime '%s' with %d threads\n" % (
          runtime_id, runtime.locality.num_threads)

    return str
  
  def load(self, system, localities):
    self.is_virtual = system.is_virtual

    # Map localities and resources
    L = [(locality.num_threads,locality) for locality in localities.values()]
    L.sort(reverse=True)

    R = [(node.num_cores,node) for node in system.values()]
    R.sort(reverse=True)

    if len(L) > len(R):
      print "Error: number of localities exceeds available resources"
      sys.exit(-1)

    for (num_threads, locality) in L:
      (num_cores, node) = R[0]

      if num_threads > num_cores:
        print "Error: no node has enough cores to host locality %s" % (locality.id)
        sys.exit(-1)
      else:
        # Create the set of local runtime instances
        runtime_id = "rts%d" % (len(self))
        self[runtime_id].load(node,locality)

      del R[0]

  def start(self, command, options):
    # Setup environment
    environment = Environment()
    environment.hpx_loglevel = options.hpx_loglevel
    environment.app_loglevel = options.app_loglevel
    environment.is_virtual = self.is_virtual

    # Setup hpx command
    program = command.split()[0]
    program_options = ' '.join(command.split()[1:])
    hpx_command = HpxCommand(program, program_options)

    hpx_command.using_gdb = options.use_gdb

    num_localities = len(self)
    hpx_command.num_localities = num_localities

    hpx_port = options.base_port + 1

    # Use 'first' runtime instance to host AGAS service
    first = self.values()[0]
    agas_ip = first.node.ip
    agas_port = options.base_port
    agas = (agas_ip, agas_port)
    hpx_command.agas_ip = agas_ip
    hpx_command.agas_port = agas_port

    # Setup local runtimes
    this_command = copy(hpx_command)
    this_command.run_agas = True
    this_command.hpx_port = hpx_port
    first.setup(program, options, this_command, environment)
    for runtime in self.values()[1:]:
      if self.is_virtual:
        hpx_port += 1
      this_command = copy(hpx_command)
      this_command.hpx_port = hpx_port
      runtime.setup(program[0:1], options, this_command, environment)

    if options.debug_mode:
      if not options.silent_mode:
        for runtime in self.values():
          print runtime
      else:
        print "Really, 'silent' mode and 'debug' mode? Hmm ..."
    else:
      if self.is_virtual:
        for runtime in self.values():
          runtime.start()
      else:
        # Clean up any hanging runs. This is necessary because a hanging HPX
        # runtime might be holding a port that we need.
        if len(options.clean) > 0:
          self.__pre_clean(options.clean)

        cleanup_file = open("cleanup.%d.sh" % (os.getpid()), 'w')
        cleanup_file.write("#!/bin/bash\n\n")
        for runtime in self.values():
          runtime.start()

          host = runtime.node.name
          pid = self.__get_remote_pid(runtime)
          cleanup_file.write("ssh %s kill -n 9 %s\n" % (host, pid))
        cleanup_file.close()

      running = self.values()
      while len(running) > 0:
        still_running = []
        for runtime in running:
          if None == runtime.process.poll():
            still_running.append(runtime)
          else:
            try:
              (stdout, stderr) = runtime.process.communicate()
              if stdout and len(stdout) > 0:
                print "Runtime '%s' stdout:\n%s" % (runtime.id, stdout)
              if stderr and len(stderr) > 0:
                print "Runtime '%s' stderr:\n%s" % (runtime.id, stderr)
              if not environment.silent_mode:
                print "Runtime '%s' quit" % (runtime.id)
            except ValueError:
              if not environment.silent_mode:
                print "Runtime '%s' quit" % (runtime.id)
        running = still_running

  def __get_remote_pid(self, runtime):
    cmd = "ps ax | grep %s " % (runtime.hpx_command.program)
    cmd += "| grep -v gdb | grep -v bash | grep -v grep "
    cmd += "| awk '//{print \$1}' "
    cmd = "ssh %s \"%s\"" % (runtime.node.name, cmd)
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    (stdout, stderr) = process.communicate()
    pid = stdout

    return pid

  def __pre_clean(self, app_name):
    for runtime in self.values():
      clean_cmd = "kill \-n 9 \`ps ax | grep %s | grep -v hpx_run | grep -v ctest | grep -v grep | awk '//{print \$1}'\`" % (app_name)
      ssh_cmd = "ssh %s \"%s\"" % (runtime.node.name, clean_cmd)

      process = subprocess.Popen(ssh_cmd, shell=True, stdout=subprocess.PIPE)
      (stdout, stderr) = process.communicate()

class LocalRuntime:
  """
  An instance of a local runtime system.
  """
  def __init__(self, id):
    self.id = id
    self.node = None
    self.locality = None
    self.hpx_command = None
    self.environment = None
    self.process = None

  def __str__(self):
    hpx_command = "(Unkown HPX command)"
    if self.hpx_command:
      hpx_command = str(self.hpx_command)

    environment = "(Unknown environment)"
    if self.environment:
      environment = str(self.environment)

    node_name = "(Unknown node name)"
    if self.node:
      node_name = self.node.name

    repr = "Local runtime instance '%s':\n" % (self.id)
    repr += "\tHPX command: %s\n" % (self.hpx_command)
    repr += "\tEnvironment: %s\n" % (self.environment)
    if self.environment and not self.environment.is_virtual:
      command = '\\"'.join(hpx_command.split('"'))
      repr += "\tSSH: ssh %s \"%s %s\"" % (node_name, environment, command)

    return repr

  def load(self, node, locality):
    self.node = node
    self.locality = locality

  def setup(self, program, options, hpx_command, environment):
    # Setup command
    self.hpx_command = hpx_command

    hpx_ip = self.node.ip
    self.hpx_command.hpx_ip = hpx_ip

    num_threads = self.locality.num_threads
    self.hpx_command.num_threads = num_threads

    # Setup environment
    self.environment = environment
    self.environment.hpx_command = self.hpx_command
    self.environment.silent_mode = options.silent_mode

  def start(self):
    if self.environment.is_virtual:
      run = str(self.hpx_command)
      self.process = subprocess.Popen(run,shell=True,stdout=subprocess.PIPE)
    else:
      prefix = self.node.name
      cmd = str(self.hpx_command)
      env = str(self.environment)

      cmd = '\\"'.join(cmd.split('"'))
      output = "2> run_%s.err > run_%s.out" % (self.id, self.id)
      run = "ssh %s \"%s %s %s\"" % (prefix, env, cmd, output)
      if not self.environment.silent_mode:
        print "$ %s" % (run)
      self.process = subprocess.Popen(run,shell=True,stdout=subprocess.PIPE)

class HpxCommand:
  """
  The HPX command to execute.
  """
  def __init__(self, program, args):
    self.program = program
    self.args = args

    self.num_localities = None
    self.num_threads = None

    self.run_agas = None

    self.agas_ip = None
    self.agas_port = None

    self.hpx_ip = None
    self.hpx_port = None

    self.using_gdb = None

  def __str__(self):
    str = ""
    if self.run_agas:
      str += "-r "
    else:
      str += "-w "

    if self.agas_ip and self.agas_port:
      str += "-a %s:%s " % (self.agas_ip, self.agas_port)

    if self.hpx_ip and self.hpx_port:
      str += "-x %s:%s " % (self.hpx_ip, self.hpx_port)

    if self.num_localities:
      str += "-l %d " % (self.num_localities)

    if self.num_threads:
      str += "-t %d " % (self.num_threads)

    if self.run_agas:
      str += self.args

    if not self.using_gdb:
      str = "%s %s" % (self.program, str)
    else:
      str = "gdb -ex \"run %s\" --batch %s" % (str, self.program)

    return str

class Environment:
  """
  The environment where the HPX command will be executed.
  """
  def __init__(self):
    self.path = "export PATH=$PATH"
    self.ld_library_path = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    self.hpx_ini = "export HPX_INI=$HPX_INI"
    self.cd_pwd = "cd $PWD"

    self.silent_mode = None
    self.is_virtual = None

    self.hpx_command = None
    self.hpx_loglevel = None
    self.app_loglevel = None

  def __str__(self):
    str = ""
    if not self.is_virtual:
      str += "%s; " % (self.path)
      str += "%s; " % (self.ld_library_path)
      str += "%s; " % (self.hpx_ini)
      str += "%s; " % (self.cd_pwd)

    if self.hpx_loglevel:
      str += "%s; " % (self.__hpx_logging())
    if self.app_loglevel:
      str += "%s; " % (self.__app_logging())

    if len(str) == 0:
      str = "(Empty)"

    return str
 
  def __hpx_logging(self):
    program = self.hpx_command.program
    num_localities = self.hpx_command.num_localities
    log_name = "%s.hpx.%d.\$[system.pid].log" % (program, num_localities)
    destination = "'file(%s)'" % (log_name)
    log_destination = "export HPX_LOGDESTINATION=%s" % (destination)

    level = self.hpx_loglevel
    log_level = "export HPX_LOGLEVEL=%d" % (level)
    
    set_hpx_logging = "%s %s" % (log_level, log_destination)

    os.environ['HPX_LOGLEVEL'] = str(level)
    os.environ['HPX_LOGDESTINATION'] = destination[1:-1].replace('\\','')

    return "%s; %s" % (log_destination, log_level)

  def __app_logging(self):
    program = self.hpx_command.program
    num_localities = self.hpx_command.num_localities
    log_name = "%s.app.%d.\$[system.pid].log" % (program, num_localities)
    destination = "'file(%s)'" % (log_name)
    log_destination = "export HPX_APP_LOGDESTINATION=%s" % (destination)

    level = self.app_loglevel
    log_level = "export HPX_APP_LOGLEVEL=%d" % (level)
    
    set_hpx_logging = "%s %s" % (log_level, log_destination)

    os.environ['HPX_APP_LOGLEVEL'] = str(level)
    os.environ['HPX_APP_LOGDESTINATION'] = destination[1:-1].replace('\\','')

    return "%s; %s" % (log_destination, log_level)

def parse_localities_spec(localities):
  localities = localities.split(':')
  
  num_localities = int(localities[0])
  num_threads = int(localities[1])

  return (num_localities, num_threads)
  
def run(options, args):
  """
  The basic idea is that we start by building a view of the system, as specified
  by a machine file or a locality specification (e.g., 2:4, for a system with
  two localities and four threads per locality). Next, we build a view of the
  localities, as specified by a locality specification or the system view.
  Finally, the distributed runtime system is built using the systems and
  localities views.

  Note, the localities view can be used to constrain the amount of system
  resources that are used, independent of a given machine file.
  """
  # Build system view
  system = System()
  if options.machine_file:
    system.load_from_machine_file(options.machine_file)
  elif options.localities:
    system.load_from_localities_spec(options.localities)
  else:
    print "Error: no machine file nor locality specification."
    sys.exit(-1)

  if not options.silent_mode:
    print system

  # Build localities view
  localities = LocalitySet()
  if options.localities:
    localities.load_from_locality_spec(options.localities)
  else:
    localities.load_from_system_view(system)

  if not options.silent_mode:
    print localities

  # Setup distributed runtime system
  distributed_runtime = DistributedRuntime()
  distributed_runtime.load(system, localities)

  if len(distributed_runtime) == 0:
    print "Err, no good pairings."
    sys.exit(-1)

  if not options.silent_mode:
    print distributed_runtime

  # Start distributed runtime system
  command = args[0]
  distributed_runtime.start(command, options)

def setup_options():
  usage = "Usage: %prog [options] command" 
  parser = OptionParser(usage=usage)
  parser.add_option("-a", "--app_logging",
                    action="store", type="int",
                    dest="app_loglevel", default=0,
                    help="Enable application logging at specified level (default 0)")
  parser.add_option("-c", "--clean",
                    action="store", type="string",
                    dest="clean", default="",
                    help="Adds support for cleaning up failed runs")
  parser.add_option("-d", "--debug",
                    action="store_true",
                    dest="debug_mode", default=False,
                    help="Put run in debug mode")
  parser.add_option("-g", "--use_gdb",
                    action="store_true",
                    dest="use_gdb", default=False,
                    help="Execute local runtimes in GDB")
  parser.add_option("-o", "--hpx_logging",
                    action="store", type="int",
                    dest="hpx_loglevel", default=0,
                    help="Enable HPX logging at specified level (default 0)")
  parser.add_option("-l", "--localities",
                    action="store", type="string",
                    dest="localities",
                    help="Specify homogeneous locality layout")
  parser.add_option("-m", "--machinefile",
                    action="store", type="string",
                    dest="machine_file",
                    help="Set resources based on machine file (e.g., PBS_NODEFILE)")
  parser.add_option("-p", "--port",
                    action="store", type="int",
                    dest="base_port", default=2222,
                    help="Set base port (default 2222)")
  parser.add_option("-s", "--shhh",
                    action="store_true",
                    dest="silent_mode", default=False,
                    help="Suppress most output")

  return parser

if __name__=="__main__":
  parser = setup_options()
  (options, args) = parser.parse_args()

  # Check for PBS_NODEFILE environment variable
  if not options.machine_file:
    options.machine_file = os.getenv("PBS_NODEFILE")

  if (len(args) > 0):
    run(options, args)
  else:
    print parser.print_help()

