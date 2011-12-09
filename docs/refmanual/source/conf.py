# -*- coding: utf-8 -*-

import sys, os

# -- General configuration -----------------------------------------------------

source_suffix = '.rst'
source_encoding = 'utf-8'
master_doc = 'index'
project = u'HPX Reference Manual'
copyright = u'2011, Hartmut Kaiser, Bryce Adelstein-Lelbach and others'
version = '0.7'
release = '0.7.0'
today_fmt = '%Y.%m.%d %H.%M.%S'
add_function_parentheses = True
pygments_style = 'sphinx'
show_authors = False

# -- Options for HTML output ---------------------------------------------------

html_theme = 'default'

# -- Options for LaTeX output --------------------------------------------------

latex_documents = [
    ( 'index'
    , 'refmanual.tex'
    , ''
    , 'Hartmut Kaiser, Bryce Adelstein-Lelbach and others'
    , 'manual'
    , False) ]

# -- Epilog for all global substitution ----------------------------------------

rst_epilog = """
.. |io_support|     replace:: :ref:`I/O support <io_support>`
.. |malloc|         replace:: :ref:`malloc allocator <linux_malloc_allocators>`
.. |env_vars|       replace:: :ref:`INI environmental variable syntax <ini_env_var_syntax>`
.. |logs|           replace:: :ref:`log levels <diagnostics_log_levels>`

.. |bsl| replace:: Boost Software License
.. _bsl: http://www.boost.org/LICENSE_1_0.txt

.. |boost| replace:: Boost C++ Libraries
.. _boost: http://boost.org

.. |gcc| replace:: GNU Compiler Collection
.. _gcc: http://gcc.gnu.org 

.. |cmake| replace:: CMake
.. _cmake: http://cmake.org

.. |jemalloc| replace:: jemalloc
.. _jemalloc: http://www.canonware.com/jemalloc

.. |google-perftools| replace:: google-perftools
.. _google-perftools: http://code.google.com/p/google-perftools 

.. |tcmalloc| replace:: tcmalloc
.. _tcmalloc: http://goog-perftools.sourceforge.net/doc/tcmalloc.html 

.. |libunwind| replace:: libunwind
.. _libunwind: http://www.nongnu.org/libunwind

.. |visualc++| replace:: Visual C++
.. _visualc++: http://msdn.microsoft.com/en-us/visualc/default.aspx

.. |elf| replace:: Executable and Linkable Format (ELF)
.. _elf: http://www.ibm.com/developerworks/power/library/pa-spec12

.. |hdf5| replace:: HDF5 Libraries
.. _hdf5: http://www.hdfgroup.org/HDF5 

.. |zlib| replace:: zlib 
.. _zlib: http://zlib.net 
"""

