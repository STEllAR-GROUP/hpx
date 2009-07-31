#!/usr/bin/python
"""hpx_log.py - an HPX logfile utility.
"""

# Copyright (c) 2009-2010 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from copy import copy
from copy import deepcopy

import sys

class HpxLog:
  """An HPX logfile"""
  def __init__(self, filename):
    self.__filename = copy(filename)
    self.__events = []

    log_file = open(filename, 'r')
    for line in log_file.readlines():
      self.__events.append(HpxLogEvent(line))

    self.__clean_counts()

  def get_events(self):
    return deepcopy(self.__events)

  def get_filename(self):
    return self.__filename

  def lines(self, sort_key='count'):
    events = [(e[sort_key],e.line()) for e in self.__events]
    events.sort()

    return [e[1] for e in events]

  def __clean_counts(self):
    """Adds zero-padding to the front of event counts."""
    count_lens = [len(e['count']) for e in self.__events]
    width = max(count_lens)

    for event in self.__events:
      event['count'] = "%%0%dd" % width % int(event['count']) 

class HpxLogEvent:
  """An event in an HPX logfile"""
  def __init__(self, log_line):
    log_line = log_line[:-1]
    self.__log_line = copy(log_line)

    self.__keys = ['locality', 'time', 'count', 'type', 'module', 'msg']
    log_items = self.__log_line.split(None, 5)

    values = [self.__clean_item(k,v) for (k,v) in zip(self.__keys,log_items)]
    self.__event = dict([[k,v] for (k,v) in zip(self.__keys, values)])

  def __clean_item(self, key, item):
    if 'locality' in key:
      return item[1:-1]
    elif 'count' in key:
      return item[1:-1]
    else:
      return item

  def __getitem__(self, key):
    return self.__event[key]

  def __setitem__(self, key, item):
    self.__event[key] = deepcopy(item)

  def __str__(self):
    line = ' '.join([self.__event[key] for key in self.__keys])

    return line

