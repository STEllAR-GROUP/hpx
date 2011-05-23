# -*- coding: utf-8 -*-

import sys, os

# -- General configuration -----------------------------------------------------

source_suffix = '.rst'
source_encoding = 'utf-8'
master_doc = 'index'
project = u'Wash\'s Notes on HPX, C++ and ParalleX'
copyright = u'2011, Bryce Lelbach (wash) <blelbach@cct.lsu.edu>'
version = '0.5'
release = '0.5.0'
today_fmt = '%Y.%m.%d_%H.%M.%S'
add_function_parentheses = True
pygments_style = 'sphinx'
show_authors = False

# -- Options for HTML output ---------------------------------------------------

html_theme = 'default'

# -- Epilog for all global substitution ----------------------------------------

rst_epilog = """
.. |amr_only|     replace:: :ref:`AMR only <linux_amr_support_libraries>`
.. |malloc|       replace:: :ref:`malloc allocator <linux_malloc_allocators>`

.. |bsl| replace:: Boost Software License
.. _bsl: http://www.boost.org/LICENSE_1_0.txt

.. |boost| replace:: Boost C++ Libraries
.. _boost: http://boost.org

.. |pathscale| replace:: EKOPath PathScale
.. _pathscale: http://www.pathscale.com

.. |icpc| replace:: Intel C++ Compiler
.. _icpc: http://software.intel.com/en-us/articles/intel-compilers

.. |clang| replace:: Clang/LLVM
.. _clang: http://github.com/lll-project

.. |gcc| replace:: GNU Compiler Collection
.. _gcc: http://gcc.gnu.org 

.. |gmake| replace:: GNU Make
.. _gmake: http://www.gnu.org/software/make

.. |libstdc++| replace:: GNU Standard C++ Library
.. _libstdc++: http://gcc.gnu.org/libstdc++

.. |eglibc| replace:: Embedded GLIBC
.. _eglibc: http://eglibc.org/home

.. |glibc| replace:: GNU C Library
.. _glibc: http://gnu.org/s/libc

.. |cmake| replace:: CMake
.. _cmake: http://cmake.org

.. |mpfr| replace:: GNU MPFR Library
.. _mpfr: http://www.mpfr.org

.. |gmp| replace:: GNU Multi-Precision Library
.. _gmp: http://gmplib.org

.. |rnpl| replace:: RNPL
.. _rnpl: http://relativity.phys.lsu.edu/postdocs/matt/software.php

.. |jemalloc| replace:: jemalloc
.. _jemalloc: http://www.canonware.com/jemalloc

.. |google-perftools| replace:: google-perftools
.. _google-perftools: http://goog-perftools.sourceforge.net

.. |tcmalloc| replace:: tcmalloc
.. _tcmalloc: http://goog-perftools.sourceforge.net/doc/tcmalloc.html 

.. |libunwind| replace:: libunwind
.. _libunwind: http://www.nongnu.org/libunwind

.. |visualc++| replace:: Visual C++
.. _visualc++: http://msdn.microsoft.com/en-us/visualc/default.aspx

.. |msbuild| replace:: MSBuild
.. _msbuild: http://msdn.microsoft.com/en-us/library/ms171452(v=vs.90).aspx

.. |elf| replace:: Executable and Linkable Format (ELF)
.. _elf: http://www.ibm.com/developerworks/power/library/pa-spec12

.. |amr| replace:: Adaptive Mesh Refinment (AMR)
.. _amr: http://en.wikipedia.org/wiki/Adaptive_mesh_refinement

.. |matt| replace:: Matthew Anderson
.. _matt: http://relativity.phys.lsu.edu/postdocs/matt/
"""

