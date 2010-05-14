#!/usr/bin/python
"""rdf.py - convert an HPX log file into an RDF graph

\tusage: python rdf.py <log-filename>
"""

# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from pyrple import bNode 
from pyrple import Graph
from pyrple import Literal
from pyrple import Node 
from pyrple import Triple

import re

RDF_URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_URI = "http://www.w3.org/2000/01/rdf-schema#"

PX_URI = "http://px.cct.lsu.edu/2010/05/px/"
HPX_URI = "http://px.cct.lsu.edu/2010/05/hpx/"

RUN_URI = "http://px.cct.lsu.edu/run/XXX/"

def uri(base, path):
  return "<%s%s>" % (base, path)

def search(event, template):
  match = template.re.search(event['msg'])
  if match:
    template.fill(event, match.groups())
    return True
  else:
    return False

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

    S = Node(uri(RUN_URI, 'component/'+self.__name))
    P = Node(uri(RDF_URI, 'type'))
    O = Node(uri(HPX_URI, 'Component'))
    triples.append(Triple(S,P,O))

    P = Node(uri(HPX_URI, 'componentId'))
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
    P = Node(uri(HPX_URI, 'numOsThreads'))
    O = Literal(str(self.__num_threads))
    t = Triple(S,P,O)

    G = Graph(triples=[t])
    return G

# Set templates to use
script_templates = [ComponentLoaded(),RunOsThreads()]

