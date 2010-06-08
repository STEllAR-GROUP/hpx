#!/usr/bin/env python

import pyrple
# print pyrple.version

LOG_NS = 'http://www.w3.org/2000/10/swap/log#'

# rawUri
# rawType
# includes
# notIncludes
# semantics
# parsedAsN3
# conclusion
# conjunction
# n3String

class BI_log_equalTo: 
    strength = 5
    def do(self, t): 
       return (t[0] == t[2])

class BI_log_notEqualTo: 
    strength = 5
    def do(self, t): 
       return (t[0] != t[2])

class BI_log_uri: 
    strength = 5
    def doObject(self, t): 
       if type(t[0]) == pyrple.URI: 
          t = t[:]
          return [t[0], t[1], pyrple.Node('"'+t[0].value+'"')]
       else: return 1

class BI_log_racine: 
    strength = 5
    def doObject(self, t): 
       if type(t[0]) == pyrple.URI: 
          t = t[:]
          if '#' in t[0].value: 
             t[2] = pyrple.Node('<'+t[0][1:(t[0].value.find('#'))+1]+'>')
          return [t[0], t[1], t[2]]
       else: return 1

class BI_log_content: 
    strength = 5
    def doObject(self, t): 
       import urllib2
       if type(t[0]) == pyrple.URI: 
          try: content = urllib2.urlopen(t[0].value).read()
          except Exception, e: content = '@@ Error getting URI: %s' % e
          return [t[0], t[1], pyrple.Node('"'+content+'"')]
       else: return 1

builtins = {}
builtins[pyrple.URI(LOG_NS + 'equalTo')] = BI_log_equalTo
builtins[pyrple.URI(LOG_NS + 'notEqualTo')] = BI_log_notEqualTo
builtins[pyrple.URI(LOG_NS + 'uri')] = BI_log_uri
builtins[pyrple.URI(LOG_NS + 'racine')] = BI_log_racine
builtins[pyrple.URI(LOG_NS + 'content')] = BI_log_content
