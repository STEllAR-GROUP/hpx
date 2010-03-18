#!/usr/bin/python
"""px_dag.py - the ParalleX computational DAG data model.

Execution
---------
- application_name
- main_thread
- threads

Thread
------
- id
- action_name
- phases

Phase
-----
- id
- thread
- locality
- transitions (out-edges)
- dependencies (in-edges)

Transistion
-----------
- target

Dependency
----------
- source

Locality
--------
- phases
"""

# Copyright (c) 2009-2010 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from copy import copy

import sys

class PxLocality:
  """A ParalleX Locality"""
  def __init__(self):
    self.__phases = ()

class PxDependency:
  """A ParalleX Dependency"""
  def __init__(self, phase):
    self.__source = phase

  def source(self):
    return self.__source

class PxTransition:
  """A ParalleX Transition"""
  def __init__(self, phase):
    self.__target = phase

  def target(self):
    return self.__target

class PxPhase:
  """A ParalleX Phase"""
  def __init__(self, id):
    self.__id = id
    self.__thread = None
    self.__locality = None
    self.__transitions = []
    self.__dependencies = []

  def __cmp__(self, other):
    if isinstance(other, PxTransition):
      return cmp(self.__id, other.target().id())
    elif isinstance(other, PxDependency):
      return cmp(self.__id, other.source().id())
    else:
      return cmp(self.__id, other.id())

  def __str__(self):
    str = "phase(%s" % (self.__id)
    str += ",%d,%d)" % (len(self.__transitions),len(self.__dependencies))

    return str

  def add_dependency_on(self, phase):
    if not phase in self.__dependencies:
      self.__dependencies.append(PxDependency(phase))

  def add_transition_to(self, phase):
    if not phase in self.__transitions:
      self.__transitions.append(PxTransition(phase))

  def id(self):
    return self.__id

  def transitions(self):
    return self.__transitions

class PxThread:
  """A ParalleX Thread"""
  def __init__(self, id, action=None):
    self.__id = id
    self.__action_name = action
    self.__phases = []

  def __cmp__(self, other):
    return cmp(self.__id, other.id())

  def __str__(self):
    str = "thread(%s" % (self.__id)
    if self.__action_name:
      str += ",\"%s\"" % (self.__action_name)
    str += ",%d)" % (len(self.__phases))

    return str

  def action_name(self):
    return self.__action_name

  def add_phase(self, phase):
    if not phase in self.__phases:
      self.__phases.append(phase)

  def id(self):
    return self.__id

  def phases(self):
    return self.__phases

  def set_action_name(self, action_name):
    self.__action_name = action_name

class PxExecution:
  """A ParalleX Computation DAG"""
  def __init__(self, application_name):
    self.__application_name = application_name
    self.__main_thread = None
    self.__threads = []

  def __str__(self):
    str = "application(\"%s\"" % (self.__application_name)
    if self.__main_thread:
      str += ",%s" % (self.__main_thread)
    str += ",%d)" % len(self.__threads)

    return str

  def add_threads(self, threads):
    self.__threads.extend([t for t in threads if not t in self.__threads])

  def set_main_thread(self, thread):
    pass
  
  def threads(self):
    return self.__threads

