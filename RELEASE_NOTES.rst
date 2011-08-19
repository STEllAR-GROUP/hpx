********************************************
 HPX 0.6.0 (alpha, codename "Velociraptor")
********************************************

Copyright (c) 1997-2011 Hartmut Kaiser, Bryce Lelbach and others

Distributed under the Boost Software License, Version 1.0. (See accompanying 
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

Major Features
--------------

    * AGAS reimplemented on top of the parcel transport layer (AGAS v2). 
    * Default AGAS version is now 2.
    * Support for distributed locality management. 
    * Distributed logging is now fully functional.
    * Distributed exception reporting is now fully functional.
    * Shutdown issues are largely mitigated.
    * Implementation of pxthread priority is complete.
    * Default thread scheduler is now local_priority.

Included Applications
---------------------

    * `dataflow`, an AMR demo from Matthew Anderson and Hartmut Kaiser.
      4 localities max.
    * `hplpx`, LU decomposition on a matrix from Daniel Kogler. SMP-only. 
    * `accumulator`, simple component example from Hartmut Kaiser. SMP-only.
    * `balancing`, load balancing demonstrations from Bryce Lelbach. SMP-only.
    * `queue`, demo of the queue LCO from Hartmut Kaiser. SMP-only.
    * `quickstart/factorial` and `quickstart/fibonacci`, future-recursive parallel
      algorithms from Bryce Lelbach. SMP-only.
    * `quickstart/hello_world`, distributed hello world example from Bryce
      Lelbach. Distributed.
    * `quickstart/quicksort`, parallel quicksort implementation from Hartmut
      Kaiser and Bryce Lelbach. SMP-only.
    * `quickstart/timed_wake`, pxthread timer demo from Bryce Lelbach. SMP-only.
    * `throttle`, shepherd thread suspension from Hartmut Kaiser and Bryce
      Lelbach. Distributed.

