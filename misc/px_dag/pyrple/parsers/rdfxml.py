#!/usr/bin/python
"""An RDF/XML Parser. Sean B. Palmer, 2003. GPL 2. Thanks to bitsko."""

import sys, re, urllib, cStringIO, xml.sax, xml.sax.handler
from urlparse import urljoin as urijoin

class Namespace(unicode): 
   def __getattr__(self, name): return self + name

rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
x = Namespace("http://www.w3.org/XML/1998/namespace")
# @@ NoNS
# @@ Unicode normalization doesn't seem to be required anymore

class Element(object): 
   def __init__(self, xmlns, name, attrs, p=None, base=None): 
      self.xmlns, self.name, self.attrs = xmlns, name, attrs or {}
      self.URI = (xmlns or '') + name
      self.base = attrs.x.get(x.base) or (p and p.base) or base or ''
      self.lang = attrs.x.get(x.lang) or (p and p.lang) or ''
      self.parent, self.children, self.text, self.subject = p, [], '', None
      self.xtext = ['<%s xmlns="%s" %r>' % (name, xmlns, attrs), '', '']

   def __getitem__(self, attr): 
      return self.attrs[attr]

   def __getitem__(self, attr): 
      return self.attrs[attr]

class Attributes(dict): 
   def __init__(self, attrs): 
      self.attrs, self.x = attrs.items(), {}
      for (xmlns, name), value in self.attrs: 
         xmlns, name = str(xmlns or ''), str(name)
         if xmlns == x: self.x[xmlns + name] = str(value)
         else: dict.__setitem__(self, xmlns + name, str(value))

   def __repr__(self): 
      return ' '.join([('xmlns:ns%s="%s" ns%s:%s="%s"' % (i, 
        self.attrs[i][0][0], i, self.attrs[i][0][1], self.attrs[i][1]), 
        '%s="%s"' % (self.attrs[i][0][1], self.attrs[i][1]))[self.attrs[i][0][0] 
        is None] for i in range(len(self.attrs))])

r_id = re.compile(r'^i([rd]+)')
r_quot = re.compile(r'([^\\])"')

class Term(str): 
   def __new__(cls, s, v): 
      a = str.__new__(cls, s)
      a.val = unicode(v)
      return a
   def __repr__(self): return self.val

class RDFParser(object): 
   def __init__(self, sink, base=None, qnames=True): 
      self.triple = sink.triple
      self.tree = []
      self.base = base or ''
      self.genID = 0
      self.qnames = qnames
      self.disallowed = [rdf.RDF, rdf.ID, rdf.about, rdf.bagID, 
           rdf.parseType, rdf.resource, rdf.nodeID, rdf.datatype, 
           rdf.li, rdf.aboutEach, rdf.aboutEachPrefix]

   def startTag(self, xmlns, name, attrs): 
      if self.tree: e = Element(xmlns, name, Attributes(attrs), self.tree[-1])
      else: e = Element(xmlns, name, Attributes(attrs), base=self.base)
      self.tree += [e]
	
   def characterData(self, chars): 
      if self.tree: 
         self.tree[-1].text += chars
         self.tree[-1].xtext[1] += chars
	
   def endTag(self, xmlns, name): 
      element = self.tree.pop()
      element.xtext[2] += '</'+element.name+'>'
      if self.tree: 
         self.tree[-1].children += [element]
         self.tree[-1].xtext[1] += ''.join(element.xtext)
      else: self.document(element)

   def uri(self, u): 
      return Term("<%s>" % u, u)

   def bNode(self, label=None): 
      if label: 
         if not label[0].isalpha(): label = 'b' + label
         return '_:' + r_id.sub('ir\g<1>', label)
      self.genID = self.genID + 1
      return Term('_:id%s' % (self.genID - 1), (self.genID - 1))

   def strlit(self, s, lang=None, dtype=None): 
      if lang and dtype: raise "ParseError", "Can't have both"
      lang = (lang and ("@" + lang) or '').lower()
      dtype = dtype and ("^^<%s>" % dtype) or ''
      try: from ntriplesg import quote
      except ImportError: return Term('"%r"' % s + lang + dtype, s)
      else: return Term('"%s"' % quote(s) + lang + dtype, s)

   def document(self, doc): 
      if doc.URI == rdf.RDF: 
         for element in doc.children: self.nodeElement(element)
      else: self.nodeElement(doc)

   def nodeElement(self, e): 
      assert e.URI not in self.disallowed, "Disallowed element used as node"

      if e.attrs.has_key(rdf.ID): 
         e.subject = self.uri(urijoin(e.base, "#" + e[rdf.ID]))
      elif e.attrs.has_key(rdf.about): 
         if (e[rdf.about] == '') and ('#' in e.base or ''): 
            base = e.base[:e.base.find('#')] # @@ urlparse oddness
            e.subject = self.uri(urijoin(base, ''))
         else: e.subject = self.uri(urijoin(e.base, e[rdf.about]))
      elif e.attrs.has_key(rdf.nodeID): e.subject = self.bNode(e[rdf.nodeID])
      elif e.subject is None: e.subject = self.bNode()

      if e.URI != rdf.Description: 
         self.triple(e.subject, self.uri(rdf.type), self.uri(e.URI))
      if e.attrs.has_key(rdf.type): 
         self.triple(e.subject, self.uri(rdf.type), self.uri(e[rdf.type]))
      for attr in e.attrs.keys(): 
         if attr not in self.disallowed + [rdf.type]: 
            objt = self.strlit(e[attr], e.lang)
            self.triple(e.subject, self.uri(attr), objt)

      for element in e.children: 
         self.propertyElt(element)

   def propertyElt(self, e): 
      if e.URI == rdf.li: 
         if not hasattr(e.parent, 'liCounter'): e.parent.liCounter = 1
         e.URI = rdf + '_' + str(e.parent.liCounter)
         e.parent.liCounter += 1

      if len(e.children) == 1 and not e.attrs.has_key(rdf.parseType): 
         self.resourcePropertyElt(e)
      elif len(e.children) == 0 and e.text: 
         self.literalPropertyElt(e)
      elif e.attrs.has_key(rdf.parseType): 
         if e[rdf.parseType] == "Resource": 
            self.parseTypeResourcePropertyElt(e)
         elif e[rdf.parseType] == "Collection": 
            self.parseTypeCollectionPropertyElt(e)
         else: self.parseTypeLiteralOrOtherPropertyElt(e)
      elif not e.text: self.emptyPropertyElt(e)

   def resourcePropertyElt(self, e): 
      n = e.children[0]
      self.nodeElement(n)
      self.triple(e.parent.subject, self.uri(e.URI), n.subject)
      if e.attrs.has_key(rdf.ID): 
         i = self.uri(urijoin(e.base, ('#' + e.attrs[rdf.ID])))
         self.reify(i, e.parent.subject, self.uri(e.URI), n.subject)

   def reify(self, r, s, p, o): 
      self.triple(r, self.uri(rdf.subject), s)
      self.triple(r, self.uri(rdf.predicate), p)
      self.triple(r, self.uri(rdf.object), o)
      self.triple(r, self.uri(rdf.type), self.uri(rdf.Statement))

   def literalPropertyElt(self, e): 
      o = self.strlit(e.text, e.lang, e.attrs.get(rdf.datatype))
      self.triple(e.parent.subject, self.uri(e.URI), o)
      if e.attrs.has_key(rdf.ID): 
         i = self.uri(urijoin(e.base, ('#' + e.attrs[rdf.ID])))
         self.reify(i, e.parent.subject, self.uri(e.URI), o)

   def parseTypeLiteralOrOtherPropertyElt(self, e): 
      o = self.strlit(e.xtext[1], e.lang, rdf.XMLLiteral)
      self.triple(e.parent.subject, self.uri(e.URI), o)
      if e.attrs.has_key(rdf.ID): 
         e.subject = i = self.uri(urijoin(e.base, ('#' + e.attrs[rdf.ID])))
         self.reify(i, e.parent.subject, self.uri(e.URI), o)

   def parseTypeResourcePropertyElt(self, e): 
      n = self.bNode()
      self.triple(e.parent.subject, self.uri(e.URI), n)
      if e.attrs.has_key(rdf.ID): 
         e.subject = i = self.uri(urijoin(e.base, ('#' + e.attrs[rdf.ID])))
         self.reify(i, e.parent.subject, self.uri(e.URI), n)
      c = Element(rdf, 'Description', e.attrs, e.parent, e.base)
      c.subject = n
      for child in e.children: 
         child.parent = c
         c.children += [child]
      self.nodeElement(c)

   def parseTypeCollectionPropertyElt(self, e): 
      for element in e.children: 
         self.nodeElement(element)
      s = [self.bNode() for f in e.children]
      if not s: 
         self.triple(e.parent.subject, self.uri(e.URI), self.uri(rdf.nil))
      else: 
         self.triple(e.parent.subject, self.uri(e.URI), s[0])
         for n in s: self.triple(n, self.uri(rdf.type), self.uri(rdf.List))
         for i in range(len(s)): 
            self.triple(s[i], self.uri(rdf.first), e.children[i].subject) 
         for i in range(len(s) - 1): 
            self.triple(s[i], self.uri(rdf.rest), s[i+1])
         self.triple(s[-1], self.uri(rdf.rest), self.uri(rdf.nil))

   def emptyPropertyElt(self, e): 
      if e.attrs.keys() in ([], [rdf.ID]): 
         r = self.strlit(e.text, e.lang) # was o
         self.triple(e.parent.subject, self.uri(e.URI), r)
      else: 
         if e.attrs.has_key(rdf.resource): 
            r = self.uri(urijoin(e.base, e[rdf.resource]))
         elif e.attrs.has_key(rdf.nodeID): r = self.bNode(e[rdf.nodeID])
         else: r = self.bNode()

         for attr in e.attrs.keys(): 
            # attrURI = attr[0] + attr[1]
            if attr not in self.disallowed: 
               if attr != rdf.type: 
                  o = self.strlit(e.attrs[attr], e.lang)
                  self.triple(r, self.uri(attr), o)
               else: self.triple(r, self.uri(rdf.type), self.uri(e.attrs[attr]))
         self.triple(e.parent.subject, self.uri(e.URI), r)
      if e.attrs.has_key(rdf.ID): 
         i = self.uri(urijoin(e.base, ('#' + e.attrs[rdf.ID])))
         self.reify(i, e.parent.subject, self.uri(e.URI), r)

class SAXRDFParser(xml.sax.handler.ContentHandler, RDFParser): 
   def __init__(self, sink, base=None): 
      RDFParser.__init__(self, sink, base=None)

   def startElementNS(self, name, qname, attribs): 
      (xmlns, name), attrs = name, dict([(attribs.getNameByQName(n), 
          attribs.getValueByQName(n)) for n in attribs.getQNames()])
      self.startTag(xmlns, name, attrs)
	
   def characters(self, chars): 
      self.characterData(chars)
	
   def endElementNS(self, name, qname): 
      xmlns, name = name
      self.endTag(xmlns, name)

DefaultHandler = SAXRDFParser

class Sink(object): 
   def __init__(self): self.result = ""
   def __str__(self): return self.result.rstrip()
   def triple(self, s, p, o): self.result += "%s %s %s .\n" % (s, p, o)
   # def write(self): return self.result.rstrip()

def parseRDF(s, base=None, sink=None): 
   sink = sink or Sink()
   parser = xml.sax.make_parser()
   parser.start_namespace_decl("xml", x)
   parser.setFeature(xml.sax.handler.feature_namespaces, 1)
   try: parser.setFeature(xml.sax.handler.feature_namespace_prefixes, 1)
   except (xml.sax._exceptions.SAXNotSupportedException, 
           xml.sax._exceptions.SAXNotRecognizedException): pass
   parser.setContentHandler(DefaultHandler(sink, base))
   parser.parse(cStringIO.StringIO(s))
   return sink

def parseURI(uri, sink=None): 
   return parseRDF(urllib.urlopen(uri).read(), base=uri, sink=sink)

if __name__=="__main__": 
   if len(sys.argv) != 2: print __doc__
   else: print parseURI(sys.argv[1])
