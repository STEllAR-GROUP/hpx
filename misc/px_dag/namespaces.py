#!/usr/bin/python
"""namespaces.py - namespace support
"""

# Copyright (c) 2010-2011 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from pyrple import Node 

def uri(base, path):
  return "<%s%s>" % (base, path)

class NS:
  def node(self, path):
    return Node(self.uri(self.URI, path))
  def item(self, uri, path):
    return Node(self.uri(self.URI+uri, path))

  def uri(self, base, path):
    return "<%s%s>" % (base, path)

class RDF(NS):
  URI = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

  def type(self): return Node(self.uri(self.URI, 'type'))

class HPX(NS):
  URI = "http://px.cct.lsu.edu/2010/05/hpx/"

  def Action(self): return self.node('Action')
  def Component(self): return self.node('Component')

  def action(self): return self.node('action')
  def componentId(self): return self.node('componentId')
  def name(self): return self.node('name')
  def numOsThreads(self): return self.node('numOsThreads')

class PX(NS):
  URI = "http://px.cct.lsu.edu/2010/05/px/"

  def Thread(self): return self.node('Thread')

class RUN(NS):
  def __init__(self, uri):
    self.URI = uri

  def action(self, path): return self.item('action/', path)
  def component(self, path): return self.item('component/', path)
  def thread(self, path): return self.item('thread/', path)

