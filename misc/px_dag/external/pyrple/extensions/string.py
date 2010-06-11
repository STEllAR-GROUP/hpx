#!/usr/bin/env python

import pyrple

STRING_NS = 'http://www.w3.org/2000/10/swap/string#'

class BI_string_notEqualTo: 
    strength = 5
    def do(self, t): 
       if (type(t[0]) != pyrple.Literal) or (type(t[2]) != pyrple.Literal): 
          return 0
       else: return (t[0].value != t[2].value)

class BI_string_startsWith: 
    strength = 5
    def do(self, t): 
       if (type(t[0]) != pyrple.Literal) or (type(t[2]) != pyrple.Literal): 
          return 0
       else: return t[0].value.startswith(t[2].value)

class BI_string_endsWith: 
    strength = 5
    def do(self, t): 
       if (type(t[0]) != pyrple.Literal) or (type(t[2]) != pyrple.Literal): 
          return 0
       else: return t[0].value.endswith(t[2].value)

builtins = {}
builtins[pyrple.URI(STRING_NS + 'notEqualTo')] = BI_string_notEqualTo
builtins[pyrple.URI(STRING_NS + 'startsWith')] = BI_string_startsWith
builtins[pyrple.URI(STRING_NS + 'endsWith')] = BI_string_endsWith
