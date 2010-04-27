#!/usr/bin/python
""" messages.py -
"""

import re

# Handle suspended thread messages
re_suspended = re.compile('suspended\(([^\.]*)\.([^\/]*)\/([^\)]*)\) P([^:]*)')

class SuspendedThread:
  """A suspended thread message"""

  def __init__(self, m_suspended):
    self.thread = 'T'+m_suspended.group(1)
    self.phase = self.thread+'p'+m_suspended.group(2)
    self.thread_gid = m_suspended.group(3)
    self.parent = 'T'+m_suspended.group(4)
    self.parent_phase = self.parent+'p'+'99' # this is just a guess

