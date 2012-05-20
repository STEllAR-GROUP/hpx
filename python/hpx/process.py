#! /usr/bin/env python 
#
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# TODO: Rename to jobs?
# TODO: More typechecking?
# TODO: Match threading.Thread interface and/or subprocess interface better?
# TODO: Better exception propagation

from threading import Thread, Lock
from time import sleep
from subprocess import Popen, STDOUT, PIPE
from types import StringType
from shlex import split
from select import epoll, EPOLLHUP

class process(object):
  _proc = None
  _error = None
  _groups = []

  def __init__(self, cmd, group=None):
    if StringType == type(cmd):
      cmd = split(cmd)

    self._proc = Popen(cmd, stderr = STDOUT, stdout = PIPE, shell = False) 

    if group is not None:
      group.add_process(self)

  def _call(self):
    # annoyingly, KeyboardInterrupts are transported to threads, while most
    # other Exceptions aren't in python
    try:
      self._proc.wait()
    except Exception, err:
      self._error = err

  def _finish(self, thread):
    timed_out = None

    # be forceful
    if thread.is_alive():
      # the thread may still be alive for a brief period after the process
      # finishes (e.g. when it is notifying groups), so we ignore any errors
      try:
        self._proc.terminate()
      except:
        pass

      thread.join()
      timed_out = True

    else:
      timed_out = False

    # if an exception happened, re-raise it here in the master thread 
    if self._error is not None:
      raise self._error

    return (timed_out, self._proc.returncode)

  def poll(self):
    return self._proc.poll()
      
  def pid(self):
    return self._proc.pid

  def wait(self, timeout=None):
    if timeout is not None:
      thread = Thread(target=self._call)
      thread.start()

      # wait for the thread and invoked process to finish
      thread.join(timeout)

      return self._finish(thread)

    else:
      return (False, self._proc.wait())

  def join(self, timeout=None):
    return self.wait(timeout)

  def read(self):
    return self._proc.stdout.read()

# modelled after Boost.Thread's boost::thread_group class
class process_group(object):
  _lock = None 
  _members = {} 
  _poller = None

  def __init__(self, *cmds):
    self._lock = Lock()
    self._poller = epoll()

    for cmd in cmds:
      self.create_process(cmd)

  def create_process(self, cmd):
    return process(cmd, self);

  def add_process(self, job):
    with self._lock:
      self._members[job._proc.stdout.fileno()] = job 
      self._poller.register(job._proc.stdout, EPOLLHUP)

  def join_all(self, timeout=None):
    if timeout is None: 
      timeout = float(-1)

    with self._lock:
      num_done = 0
  
      while True:
        for fd, flags in self._poller.poll(timeout=timeout):
          self._poller.unregister(fd)
          num_done += 1
  
        if len(self._members) == num_done:
          break

def join_all(*tasks, **keys):
  def flatten(items):
    result = []

    for element in items:
      if hasattr(element, "__iter__"):
        result.extend(flatten(el))

      else:
        if not isinstance(element, process):
          raise TypeError( "'%s' is not an instance of 'hpx.process'"
                         % str(element))

        result.append(element)

    return result

  tasks = flatten(tasks)

  pg = process_group()

  for task in tasks:
    pg.add_process(task)

  pg.join_all(keys['timeout'])

