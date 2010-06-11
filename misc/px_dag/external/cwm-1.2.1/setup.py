#!/usr/bin/env python
# borrowing liberally from Eikeon's setup.py for rdflib
# an attempt at a setup.py installer for Cwm/SWAP
# version: $Id: setup.py,v 1.23 2007/12/16 00:22:59 syosi Exp $
# by Dan Brickley <danbri@w3.org>
#
# STATUS: this file now works
# much of what danbri typed I honestly do not know what it is trying to do. I suspect the answer is
# ``not much''
#
# DO NOT run this file directly! Use make dist_tarball instead.
# 
# notes:
# http://esw.w3.org/t/view/ESW/CwmTips


from distutils.sysconfig import get_python_lib
from os import rename
from os.path import join, exists
from time import time

lib_dir = get_python_lib()
swap_dir = join(lib_dir, "swap")
print "swap dir: "+swap_dir


##if exists(swap_dir):
##    backup = "%s-%s" % (swap_dir, int(time()))
##    print "Renaming previously installed swap to: \n  %s" % backup
##    rename(swap_dir, backup)


# Install SWAP
from distutils.core import setup
#from swap import __version__
__version__='1.2.1'
setup(
    name = 'cwm',
    version = __version__,
    description = "Semantic Web Area for Play",
    author = "TimBL, Dan Connolly and contributors",
    author_email = "timbl@w3.org",
    maintainer = "Tim Berners-Lee",
    maintainer_email = "timbl@w3.org",
    url = "http://www.w3.org/2000/10/swap/",
    package_dir = {'swap': 'swap'},
    packages = ['swap'],
#    py_modules = ['cwm', 'delta', 'cant'],
    scripts = ['cwm', 'delta', 'cant.py'],
   )
    # todo, figure out which other modules are in public APIs
    # --danbri



#,'swap.cwm','swap.RDFSink','swap.llyn'],
#		'swap.RDFSink',
#		'swap.llyn'],
#    packages = ['swap.cwm',
#		'swap.RDFSink',
#		'swap.llyn'],
#    package_dir = {'': 'swap'},
