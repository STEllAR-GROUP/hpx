*******************
 Malloc Allocators 
*******************

:author: Bryce Lelbach
:organization: LSU

::
  Copyright (C) 2011 Bryce Lelbach

  Distributed under the Boost Software License, Version 1.0. (See accompanying 
  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

--------------------------------------------------------------------------------

HPX supports the use of alternative malloc implementations. On Linux systems,
an alternative malloc allocator is used by default if one is available. The
default is `tcmalloc`_, and the first fallback is `jemalloc`_.

For some HPX applications (ones in which allocations of 64-bit objects or
smaller are predominant), jemalloc may perform better than tcmalloc. In all
other cases, tcmalloc will perform better than jemalloc. The system allocator
is almost always the slowest option available. Benchmarks of all three allocators
can be found in $HPX_SOURCE/tests/performance/allocator.

To specify the root of your tcmalloc and/or jemalloc installation, define the 
CMake variables TCMALLOC_ROOT and/or JEMALLOC_ROOT to point the appropriate
paths. 

To change the malloc allocator, define the HPX_MALLOC CMake variable to the name
of your preferred allocator. Valid options are "tcmalloc", "jemalloc" and
"system".

.. _jemalloc: http://www.canonware.com/jemalloc
.. _tcmalloc: http://goog-perftools.sourceforge.net/doc/tcmalloc.html 

