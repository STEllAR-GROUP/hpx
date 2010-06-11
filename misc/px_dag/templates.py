#!/usr/bin/python
"""templates.py - 
"""

# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from namespaces import HPX
from namespaces import RDF
from namespaces import RUN
from namespaces import PX

from pyrple import bNode 
from pyrple import Graph
from pyrple import Literal
from pyrple import Node 
from pyrple import Triple

import re

RDF = RDF()
HPX = HPX()
PX = PX()

RUN = RUN("_:")

#####

class Template:
  def fill(self, event, groups):
    pass

  def in_english(self): 
    return ''

  def as_rdf(self): 
    return []

class ComponentLoaded(Template):
  def __init__(self):
    self.__name = None
    self.__id = None

    regex = 'dynamic loading succeeded: ([^:]+): ([^:]+): [^\[]*\[([^\]]+)\]'
    self.re = re.compile(regex)

  def fill(self, event, groups):
    self.__name = groups[1]
    self.__id = groups[2]

  def in_english(self):
    str = "Component '%s' has id %s." % (self.__name, self.__id)
    return str

  def as_rdf(self):
    triples = []

    S = RUN.component(self.__name)
    P = RDF.type()
    O = HPX.Component()
    triples.append(Triple(S,P,O))

    P = HPX.componentId()
    O = Literal(self.__id)
    triples.append(Triple(S,P,O))
    
    G = Graph(triples=triples)
    return G

class RunOsThreads(Template):
  def __init__(self):
    self.__num_threads = -1

    regex = 'run: creating (\d+) OS thread\(s\)'
    self.re = re.compile(regex)

  def fill(self, event, groups):
    self.__num_threads = int(groups[0])

  def in_english(self):
    str = "This run is using %d OS threads." % (self.__num_threads)
    return str

  def as_rdf(self):
    S = bNode('run')
    P = HPX.numOsThreads()
    O = Literal(str(self.__num_threads))
    t = Triple(S,P,O)

    G = Graph(triples=[t])
    return G

class ThreadThread(Template):
  def __init__(self):
    self.__gid = None
    self.__action_name = None

    regex = 'thread::thread\(([^\)]+)\), description\(([^\)]+)\)'
    self.re = re.compile(regex)
  
  def fill(self, event, groups):
    self.__gid = groups[0]
    self.__action_name = groups[1]

  def in_english(self):
    str = "Action '%s' instantiated as thread '%s'." % (self.__action_name, self.__gid)
    return str

  def as_rdf(self):
    action = RUN.action(self.__action_name)
    action_name = Literal(self.__action_name)
    thread = RUN.px_thread(self.__gid)

    triples = [Triple(action, RDF.type(), HPX.Action()),
               Triple(action, HPX.name(), action_name),
               Triple(thread, RDF.type(), PX.Thread()),
               Triple(thread, HPX.action(), action)]

    G = Graph(triples=triples)
    return G

class NumHpxThreads(Template):
  def __init__(self):
    self.__id = None
    self.__num_hpx_threads = None

    regex = 'tfunc\(([^\)]+)\): end, executed (\d+) HPX threads'
    self.re = re.compile(regex)
  
  def fill(self, event, groups):
    self.__locality = event['thread'][0]
    self.__id = groups[0]
    self.__num_hpx_threads = int(groups[1])

  def in_english(self):
    str = "HPX instance '%s' instantiated %d threads." % (self.__id, self.__num_hpx_threads)
    return str

  def as_rdf(self):
    hpx_thread = RUN.hpx_thread(self.__locality+'/'+ self.__id)
    id = Literal(self.__id)
    num_hpx_threads = Literal(str(self.__num_hpx_threads))

    triples = [Triple(hpx_thread, HPX.name(), id),
               Triple(hpx_thread, HPX.numHpxThreads(), num_hpx_threads)]
  
    G = Graph(triples=triples)
    return G

# Templates for lines with no semantic value

class ConnectionCache(Template):
  regex = 'connection_cache: '
  re = re.compile(regex)

class Tfunc(Template):
  regex = 'tfunc\([^\)]*\): '
  re = re.compile(regex)

