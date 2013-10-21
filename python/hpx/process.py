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

from sys import float_info, platform
from threading import Thread, Lock
from time import sleep, time
from subprocess import Popen, STDOUT, PIPE
from types import StringType
from shlex import split
from signal import SIGKILL
from os import kill, killpg, waitpid, WNOHANG
from platform import system
from Queue import Queue, Empty
from errno import ESRCH
# TODO: implement for Windows

OS_MAC = False
OS_LIN = False

if platform.startswith('darwin'):
  OS_MAC = True
  from select import kqueue, kevent
  from select import KQ_FILTER_READ, KQ_EV_ADD, KQ_EV_DELETE, KQ_NOTE_LOWAT

if platform.startswith('linux'):
  OS_LIN = True
  from select import epoll, EPOLLHUP

def kill_process_tree(parent_pid, signal=SIGKILL):
  def find_process_tree(pid):
    cmd = ""

    if OS_MAC:
      cmd = "ps -o pid,ppid -ax | egrep '%d$' | awk '{print $1}'" % pid
    else: 
      cmd = "ps -o pid --ppid %d --noheaders" % pid

    ps_command = Popen(cmd, shell=True, stdout=PIPE)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()

    if 0 == ps_command.wait():
      list = [pid]

      for child_pid in ps_output.split("\n")[:-1]:
        list = list + find_process_tree(int(child_pid))

      return list

    else:
      return [pid]

  r = True

  for pid in find_process_tree(parent_pid):
    try:
      kill(int(pid), signal)
    except OSError, err:
      if ESRCH != err.errno:
        raise err
      else:
        r = False

  return r

class process(object):
  _proc = None
  _error = None
  _groups = None
  _timed_out = None

  def __init__(self, cmd, group=None):
    if StringType == type(cmd):
      cmd = split(cmd)

    self._proc = Popen(cmd, stderr=STDOUT, stdout=PIPE, shell=False)

    self._error = None
    self._groups = []
    self._timed_out = False

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
    # be forceful
    if thread.is_alive():
      # the thread may still be alive for a brief period after the process
      # finishes (e.g. when it is notifying groups), so we ignore any errors
      self.terminate()
      thread.join()

      self._timed_out = True

    # if an exception happened, re-raise it here in the master thread
    if self._error is not None:
      raise self._error

    return (self._timed_out, self._proc.returncode)

  def terminate(self):
    return kill_process_tree(self.pid())

  def poll(self):
    return self._proc.poll()

  def pid(self):
    return self._proc.pid

  def fileno(self):
    return self._proc.stdout.fileno()

  def timed_out(self):
    return self._timed_out

  def wait(self, timeout=None):
    if timeout is not None:
      thread = Thread(target=self._call)
      thread.start()

      # wait for the thread and invoked process to finish
      thread.join(timeout)

      return self._finish(thread)

    else:
      return (self._timed_out, self._proc.wait())

  def join(self, timeout=None):
    return self.wait(timeout)

  def read(self, timeout=None):
    read_queue = Queue()

    def enqueue_output():
      for block in iter(self._proc.stdout.read, b''):
        read_queue.put(block)

      read_queue.put('')

    thread = Thread(target=enqueue_output)
    thread.daemon = True
    thread.start()

    output = ''

    try:
      started = time()

      while timeout is None or not float_info.epsilon > timeout:
        s = read_queue.get(timeout=timeout)

        if s:
          output += s
        else:
          return output

        if not timeout is None:
          timeout -= (time() - started)
    except Empty:
      return output

# modelled after Boost.Thread's boost::thread_group class
class process_group(object):
  _lock = None
  _members = None
  _poller = None

  def __init__(self, *cmds):
    self._lock = Lock()
    self._members = {}
    if OS_MAC:
      self._poller = kqueue()
    if OS_LIN:
      self._poller = epoll()

    for cmd in cmds:
      self.create_process(cmd)

  def create_process(self, cmd):
    return process(cmd, self)

  def add_process(self, job):
    with self._lock:
      self._members[job.fileno()] = job
      if OS_MAC:
        self._poller.control([kevent(job._proc.stdout,
          KQ_FILTER_READ, KQ_EV_ADD, KQ_NOTE_LOWAT, 0)], 0)
      if OS_LIN:
        self._poller.register(job._proc.stdout, EPOLLHUP)

  def join_all(self, timeout=None, callback=None):
    with self._lock:
      not_done = self._members.copy()

      started = time()

      while timeout is None or not float_info.epsilon > timeout:
          if OS_MAC:

            if timeout == None:
              timeout=-1.0

            ready = self._poller.control(None,1,timeout)
          if OS_LIN:
            ready = self._poller.poll(timeout=-1.0 if timeout is None else timeout)

          if not timeout is None:
            timeout -= (time() - started)

          if OS_MAC:
            for fd in ready:
              fd = fd.ident
              self._poller.control([kevent(fd, KQ_FILTER_READ, KQ_EV_DELETE)], 0)
              not_done.pop(fd)

              if callable(callback):
                callback(fd, self._members[fd])

          if OS_LIN:
            for fd, flags in ready:
              self._poller.unregister(fd)
              not_done.pop(fd)

              if callable(callback):
                callback(fd, self._members[fd])

          if 0 == len(not_done):
            return

      # some of the jobs are not done, we'll have to forcefully stop them
      for fd in not_done:
        if self._members[fd].terminate():
          self._members[fd]._timed_out = True

        if callable(callback):
          callback(fd, self._members[fd])

  def read_all(self, timeout=None, callback=None):
    output = {}

    def read_callback(fd, job):
      output[fd] = job.read(0.5)

      if callable(callback):
        callback(fd, job, output[fd])

    self.join_all(timeout, read_callback)

    return output

  def terminate_all(self, callback=None):
    with self._lock:
      for (fd, job) in self._members.iteritems():
        if job.terminate():
          if callable(callback):
            callback(fd, job)

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

  pg.join_all(keys['timeout'], keys['callback'])

def read_all(*tasks, **keys):
  output = {}

  callback = keys['callback']

  def read_callback(fd, job):
    output[fd] = job.read()

    if callable(callback):
      callback(fd, job, output[fd])

  keys['callback'] = read_callback

  join_all(*tasks, **keys)

  return output

