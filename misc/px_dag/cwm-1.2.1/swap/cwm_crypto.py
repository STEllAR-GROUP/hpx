#!/usr/bin/python 
"""
Cryptographic Built-Ins for CWM/Llyn

The continuing story of cryptographic builtins.

cf. http://www.w3.org/2000/10/swap/cwm.py
& http://www.amk.ca/python/writing/pycrypt/node16.html
"""

__author__ = 'Sean B. Palmer'
__cvsid__ = '$Id: cwm_crypto.py,v 1.11 2005/07/21 15:22:59 syosi Exp $'
__version__ = '$Revision: 1.11 $'

import md5, sha, binascii, quopri, base64
from term import Function, ReverseFunction, LightBuiltIn

USE_PKC = 1

if USE_PKC:
    try:
        import Crypto.Util.randpool as randpool
        import Crypto.PublicKey.RSA as RSA
    except ImportError:
        USE_PKC = 0
#        'we failed')

# Some stuff that we need to know about

CRYPTO_NS_URI = 'http://www.w3.org/2000/10/swap/crypto#'

# A debugging function...

def formatObject(obj): 
   """Print the various bits found within a key (works on any object)."""
   if ' ' in repr(obj): result = repr(obj)[1:].split(' ')[0]+'\n'
   else: result = '\n'
   for n in dir(obj): result += str(n)+' '+str(getattr(obj, n))+'\n'
   return '[[[%s]]]' % result

# Functions for constructing keys, and formatting them for use in text

def newKey(e, n, d=None, p=None, q=None): 
   """Create a new key."""
   key = RSA.RSAobj() # Create a new empty RSA Key
   key.e, key.n = e, n # Feed it the ee and modulus
   if d is not None: 
      key.d, key.p, key.q = d, p, q
      return key
   else: return key.publickey() # Return the public key variant

def keyToQuo(key, joi='\n\n'): 
   """Returns a quoted printable version of a key - ee then m.
   Leading and trailing whitespace is allowed; stripped by quoToKey."""
   e, n = str(key.e), str(key.n) # Convert the ee and mod to strings
   if key.has_private():
      d, p, q = str(key.d), str(key.p), str(key.q)
      strkey = base64.encodestring(joi.join([e, n, d, p, q]))
   else:
      strkey = base64.encodestring('%s%s%s' % (e, joi, n))
   return '\n'+quopri.encodestring(strkey).strip()+'\n'

def quoToKey(strkey, spl='\n\n'): 
   """Returns a key from quopri (ee then m) version of a key."""
   bunc = base64.decodestring(quopri.decodestring(strkey.strip()))
   bits = bunc.split(spl)
   if len(bits) == 2: return newKey(long(bits[0]), long(bits[1]))
   else: 
      e, n, d, p, q = bits
      return newKey(long(e), long(n), long(d), long(p), long(q))

# Signature encoding and decoding

def baseEncode(s): 
   s = base64.encodestring(s)
   return '\n'+quopri.encodestring(s).strip()+'\n'

def baseDecode(s): 
   s = quopri.decodestring(s.strip())
   return base64.decodestring(s)

# Decimal to binary

def decToBin(i): # int to string
    result = ''
    while i > 0: 
        d = i % 2
        result = str(d)+result
        i /= 2
    return result

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# C R Y P T O G R A H P I C   B U I L T - I N s
#
# At the moment, we only have built-ins that can gague the hash values of 
# strings. It may be cool to have built ins that can give the hash value 
# of the content of a work, too, although you can do that with log:content.
#
#   Light Built-in classes

#  Hash Constructors - light built-ins

class BI_md5(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        m = md5.new(subj_py).digest() 
        return  binascii.hexlify(m)

class BI_sha(LightBuiltIn, Function):
    def evaluateObject(self, subj_py): 
        m = sha.new(subj_py).digest() 
        return binascii.hexlify(m)

# Create a new RSA key

class BI_keyLength(LightBuiltIn, Function, ReverseFunction):
   def __init__(self, resource, fragid): 
      LightBuiltIn.__init__(self, resource, fragid)
      Function.__init__(self)
      ReverseFunction.__init__(self)
      self.do = 1

   def evaluateSubject(self,  obj_py): 
      """Generates an RSA keypair, and spews it out as plain text.
         Has the limitation that it will *only* ever let you generate 
         one key pair (per iteration), in order to work around a bug."""
      if self.do: 
         randfunc, self.do = randpool.RandomPool(int(obj_py)), 0
         RSAKey = RSA.generate(int(obj_py), randfunc.get_bytes)
         TextKey = keyToQuo(RSAKey)
         if TextKey != 'N.': return TextKey

   def evaluateObject(self,  subj_py): 
      RSAKey = quoToKey(subj_py)
      return str(len(decToBin(RSAKey.n))) # @@ not integer?

class BI_sign(LightBuiltIn, Function): 
   def evaluateObject(self, subj_py): 
      """Sign a hash with a key, and get a signature back."""
      import time
      hash, keypair = subj_py
      RSAKey = quoToKey(keypair)
      signature = RSAKey.sign(hash, str(time.time())) # sign the hash with the key
      return baseEncode(str(signature[0]))

class BI_verify(LightBuiltIn): 
   def evaluate(self, subj_py, obj_py): 
      """Verify a hash/signature."""
      keypair, (hash, signature) = subj_py, obj_py
      hash = hash.encode('ascii')
      RSAKey = quoToKey(keypair) # Dequote the key
      signature = (long(baseDecode(signature)),) # convert the signature back
      return RSAKey.verify(hash, signature)

class BI_verifyBoolean(LightBuiltIn, Function): 
   def evaluateObject(self, subj_py): 
      """Verify a hash/signature."""
      keypair, hash, signature = subj_py
      hash = hash.encode('ascii')
      RSAKey = quoToKey(keypair) # Dequote the key
      signature = (long(baseDecode(signature)),)
      result = RSAKey.verify(hash, signature)
      return str(result)

class BI_publicKey(LightBuiltIn, Function): 
   def evaluateObject(self, subj_py): 
      """Generate a quopri public key from a keypair."""
      keypair = quoToKey(subj_py) # Dequote the key
      publickey = keypair.publickey() # Get the public key
      return keyToQuo(publickey)

#  Register the string built-ins with the store

def register(store):
   str = store.symbol(CRYPTO_NS_URI[:-1])
   str.internFrag('md5', BI_md5)
   str.internFrag('sha', BI_sha)
   if USE_PKC: 
      str.internFrag('keyLength', BI_keyLength)
      str.internFrag('sign', BI_sign)
      str.internFrag('verify', BI_verify)
      str.internFrag('verifyBoolean', BI_verifyBoolean)
      str.internFrag('publicKey', BI_publicKey)

if __name__=="__main__": 
   print __doc__.strip()
