**************
 Building HPX
**************

:authors: Hartmut Kaiser, Chirag Dekate, Bryce Lelbach
:organization: LSU

Distributed under the Boost Software License, Version 1.0. (See accompanying 
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

Prerequisites
=============

Build System
------------

+------------------------------+---------------------+-----------------+
| Name                         | Recommended Version | Min/Max Version |
+------------------------------+---------------------+-----------------+
| `CMake`_ (POSIX, Windows)    | 2.8.2               | 2.6.2 - 2.8.*   |
+------------------------------+---------------------+-----------------+
| `Visual Studio`_ (Windows)   | 2010                | 2005 - 2010     |
+------------------------------+---------------------+-----------------+

.. _CMake: http://cmake.org
.. _Visual Studio: http://www.microsoft.com/visualstudio/en-us/

Compilers
---------

+------------------------------+---------------------+-----------------+
| Name                         | Recommended Version | Min/Max Version |
+------------------------------+---------------------+-----------------+
| `Intel C++`_ (POSIX)         | 12.0 (XE 2011)      | 11.1 - 12.0     |
+------------------------------+---------------------+-----------------+
| `GNU GCC`_ (POSIX)           | 4.4.5               | 4.1.2 - 4.6.*   |
+------------------------------+---------------------+-----------------+
| `Visual C++`_ (Windows)      | 2010                | 2005 - 2010     |
+------------------------------+---------------------+-----------------+
| `Clang/LLVM`_ (POSIX)        | 2.9                 | 2.9             |
+------------------------------+---------------------+-----------------+

.. _Intel C++: http://msdn.microsoft.com/en-us/visualc/default.aspx
.. _GNU GCC: http://gcc.gnu.org 
.. _Visual C++: http://software.intel.com/en-us/articles/intel-compilers
.. _Clang/LLVM: http://clang.llvm.org

Libraries
---------

+------------------------------+---------------------+-----------------+
| Name                         | Recommended Version | Min/Max Version |
+------------------------------+---------------------+-----------------+
| `Boost`_ (POSIX, Windows)    | SVN                 | 1.40.0 - SVN    |
+------------------------------+---------------------+-----------------+
| `EGLIBC`_ (Debian, Ubuntu)   | 2.11.*              | 2.7.* - 2.13.*  |
+------------------------------+---------------------+-----------------+
| `glibc`_ (Redhat, Fedora)    | 2.11.*              | 2.7.* - 2.13.*  |
+------------------------------+---------------------+-----------------+
| `libstdc++`_ (POSIX)         | 4.4.5               | 4.1.2 - 4.6.*   |
+------------------------------+---------------------+-----------------+

.. _Boost: http://boost.org
.. _EGLIBC: http://eglibc.org/home
.. _glibc: http://gnu.org/s/libc
.. _libstdc++: http://gnu.org/s/libc

Optional Libraries
------------------

+------------------------------+---------------------+-----------------+
| Name                         | Recommended Version | Min/Max Version |
+------------------------------+---------------------+-----------------+
| `MPFR`_ (POSIX)              | 3.0.0               | 3.0.0 - SVN     |
+------------------------------+---------------------+-----------------+
| `GMP`_ (POSIX)               | 4.2.3               | 4.1.0 - 5.0.1   |
+------------------------------+---------------------+-----------------+
| `SDF`_ (POSIX)               | N/A                 | N/A             |
+------------------------------+---------------------+-----------------+

.. _MPFR: http://www.mpfr.org
.. _GMP: http://gmplib.org
.. _SDF: http://relativity.phys.lsu.edu/postdocs/matt/software.php

POSIX Build Instructions
========================

0) Download the latest Subversion release of HPX:::

  $ svn co https://svn.cct.lsu.edu/repos/projects/parallex/trunk/hpx hpx

1) Invoke CMake, setting the BOOST_ROOT cmake variable to point to the root
   of a Boost source tree or the root of a system-installed Boost (example
   CMake configurations can be found in $HPX_ROOT/build). HPX requires an out
   of tree build. You can set the CMake variable CMAKE_PREFIX to specify an
   install location for HPX. By default, HPX installs to /usr/local/.::

  $ cd hpx
  $ mkdir my_hpx_build
  $ cd my_hpx_build
  $ cmake -DBOOST_ROOT=/path/to/boost ..

2) Invoke GNU make. If you are on a box with multiple cores (very likely),
   add the -jN flag to your make invocation, where N is the number of nodes
   on your machine plus one.::

  $ make -j3
 
3) To complete the build and install HPX:::

  $ make install

If you have difficulty in compiling the code, please email 
the `HPX developers <gopx@cct.lsu.edu>`_.

Malloc Allocator
================

HPX supports the use of alternative malloc implementations. Currently, HPX ships
with two allocators, `jemalloc`_ and `nedmalloc`_. In the default configuration,
jemalloc is used for POSIX systems, and the system allocator is used on Windows.
To switch allocators, set the CMake variable HPX_MALLOC to the name of your
preferred allocator. Set the variable to "system" to use your default system
allocator on POSIX.

.. _jemalloc: http://www.canonware.com/jemalloc/
.. _nedmalloc: http://www.nedprod.com/programs/portable/nedmalloc/

