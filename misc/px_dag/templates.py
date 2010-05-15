#!/usr/bin/python
"""rdf.py - convert an HPX log file into an RDF graph

\tusage: python rdf.py <log-filename>
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

class ComponentLoaded():
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
    O = HPX.Components()
    triples.append(Triple(S,P,O))

    P = HPX.componentId()
    O = Literal(self.__id)
    triples.append(Triple(S,P,O))
    
    G = Graph(triples=triples)
    return G

class RunOsThreads():
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

# Set templates to use
script_templates = [ComponentLoaded(),RunOsThreads()]

