"""filter.py - standard filter
"""

import templates

# Set templates to use
script_templates = [
  templates.ComponentLoaded(),
  templates.RunOsThreads(),
  templates.ThreadThread(),
  templates.NumHpxThreads(),
]

# templates with no semantic value: useful for debugging or filtering
script_templates += [
  templates.ConnectionCache(),
  #templates.Tfunc(),
]
