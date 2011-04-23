***********************
 Building HPX on Linux
***********************

:author: Bryce Lelbach
:organization: LSU

Distributed under the Boost Software License, Version 1.0. (See accompanying 
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

Prerequisites
=============

Compilers
---------

+------------------------------+---------------------+-----------------+
| Name                         | Recommended Version | Min/Max Version |
+------------------------------+---------------------+-----------------+
| `EKOPath PathScale`_         | 4.0.9               | 4.0.0 - 4.0.9   |
+------------------------------+---------------------+-----------------+
| `Intel C++`_                 | 12.0 (XE 2011)      | 11.1 - 12.0     |
+------------------------------+---------------------+-----------------+
| `Clang/LLVM`_                | 3.0 (lll)           | 2.9 - 3.0 (lll) |
+------------------------------+---------------------+-----------------+
| `GNU GCC`_                   | 4.4.4               | 4.4.2 - 4.4.5   |
+------------------------------+---------------------+-----------------+

.. _EKOPath PathScale: http://www.pathscale.com
.. _Intel C++: http://software.intel.com/en-us/articles/intel-compilers
.. _Clang/LLVM: http://github.com/lll-project
.. _GNU GCC: http://gcc.gnu.org 

Build System
------------

+------------------------------+---------------------+-----------------+
| Name                         | Recommended Version | Min/Max Version |
+------------------------------+---------------------+-----------------+
| `CMake`_                     | 2.6.4               | 2.6.4 - 2.8.*   |
+------------------------------+---------------------+-----------------+
| `GNU Make`_                  | 3.81                | 3.80 - 3.82     |
+------------------------------+---------------------+-----------------+

.. _CMake: http://cmake.org
.. _GNU Make: http://www.gnu.org/software/make

Required Libraries
------------------

+---------------------------------+---------------------+-----------------+
| Name                            | Recommended Version | Min/Max Version |
+---------------------------------+---------------------+-----------------+
| `Boost`_                        | 1.45.0              | 1.43.0 - SVN    |
+---------------------------------+---------------------+-----------------+
| `EGLIBC`_ (Debian, Ubuntu)      | 2.11.*              | 2.7.* - 2.13.*  |
+---------------------------------+---------------------+-----------------+
| `glibc`_ (Redhat)               | 2.11.*              | 2.7.* - 2.13.*  |
+---------------------------------+---------------------+-----------------+
| `GNU libstdc++`_                | 4.4.4               | 4.4.2 - 4.4.5   |
+---------------------------------+---------------------+-----------------+

.. _Boost: http://boost.org
.. _EGLIBC: http://eglibc.org/home
.. _glibc: http://gnu.org/s/libc
.. _GNU libstdc++: http://gcc.gnu.org/libstdc++

Optional Libraries
------------------

+---------------------------------+---------------------+-----------------+
| Name                            | Recommended Version | Min/Max Version |
+---------------------------------+---------------------+-----------------+
| `MPFR`_                         | 3.0.0               | 3.0.0 - SVN     |
+---------------------------------+---------------------+-----------------+
| `GMP`_                          | 4.2.3               | 4.1.0 - 5.0.1   |
+---------------------------------+---------------------+-----------------+
| `SDF`_                          | N/A                 | N/A             |
+---------------------------------+---------------------+-----------------+
| `jemalloc`_                     | 2.1.2               | 2.1.0 - GIT     |
+---------------------------------+---------------------+-----------------+
| `google-perftools (tcmalloc)`_  | 1.7.1               | 1.7.1 - SVN     |
+---------------------------------+---------------------+-----------------+
| `libunwind`_                    | 0.99                | 0.97 - GIT      |
+---------------------------------+---------------------+-----------------+

.. _MPFR: http://www.mpfr.org
.. _GMP: http://gmplib.org
.. _SDF: http://relativity.phys.lsu.edu/postdocs/matt/software.php
.. _jemalloc: http://www.canonware.com/jemalloc
.. _google-perftools (tcmalloc): http://goog-perftools.sourceforge.net
.. _libunwind: http://www.nongnu.org/libunwind

Build Instructions
==================

0) Download the latest Subversion release of HPX:::

  $ svn co https://svn.cct.lsu.edu/repos/projects/parallex/trunk/hpx hpx

1) Create a build directory. HPX requires an out-of-tree build. This means you
   will be unable to run CMake in the HPX source tree.::
  
  $ cd hpx
  $ mkdir my_hpx_build
  $ cd my_hpx_build

2) Invoke CMake from your build directory, pointing the CMake driver to the root
   of your HPX source tree.::

  $ cmake [CMake variable definitions] /path/to/source/tree 

3) Invoke GNU make. If you are on a box with multiple cores (very likely),
   add the -jN flag to your make invocation, where N is the number of nodes
   on your machine plus one.::

  $ gmake -j49
 
4) To complete the build and install HPX:::

  $ gmake install

