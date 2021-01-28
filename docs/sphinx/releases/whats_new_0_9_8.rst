..
    Copyright (C) 2007-2018 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. include:: <isonum.txt>

.. _hpx_0_9_8:

===========================
|hpx| V0.9.8 (Mar 24, 2014)
===========================

We have had over 800 commits since the last release and we have closed over 65
tickets (bugs, feature requests, etc.).

With the changes below, |hpx| is once again leading the charge of a whole new
era of computation. By intrinsically breaking down and synchronizing the work to
be done, |hpx| insures that application developers will no longer have to fret
about where a segment of code executes. That allows coders to focus their time
and energy to understanding the data dependencies of their algorithms and
thereby the core obstacles to an efficient code. Here are some of the advantages
of using |hpx|:

* |hpx| is solidly rooted in a sophisticated theoretical execution model --
  ParalleX
* |hpx| exposes an API fully conforming to the C++11 and the draft C++14
  standards, extended and applied to distributed computing. Everything
  programmers know about the concurrency primitives of the standard C++ library
  is still valid in the context of |hpx|.
* It provides a competitive, high performance implementation of modern,
  future-proof ideas which gives an smooth migration path from today's
  mainstream techniques
* There is no need for the programmer to worry about lower level parallelization
  paradigms like threads or message passing; no need to understand pthreads,
  MPI, OpenMP, or Windows threads, etc.
* There is no need to think about different types of parallelism such as tasks,
  pipelines, or fork-join, task or data parallelism.
* The same source of your program compiles and runs on Linux, BlueGene/Q, Mac OS
  X, Windows, and Android.
* The same code runs on shared memory multi-core systems and supercomputers, on
  handheld devices and Intel\ |reg| Xeon Phi\ |trade| accelerators, or a
  heterogeneous mix of those.

General changes
===============

* A major API breaking change for this release was introduced by implementing
  ``hpx::future`` and ``hpx::shared_future`` fully in conformance with the
  |cpp11|_. While ``hpx::shared_future`` is new and will not create any
  compatibility problems, we revised the interface and implementation of the
  existing ``hpx::future``. For more details please see the `mailing list
  archive
  <http://mail.cct.lsu.edu/pipermail/hpx-users/2014-January/000141.html>`_. To
  avoid any incompatibilities for existing code we named the type which
  implements the ``std::future`` interface as ``hpx::unique_future``. For the
  next release this will be renamed to ``hpx::future``, making it full
  conforming to |cpp11|_.
* A large part of the code base of |hpx| has been refactored and partially
  re-implemented. The main changes were related to

  * The threading subsystem: these changes significantly reduce the amount of
    overheads caused by the schedulers, improve the modularity of the code
    base, and extend the variety of available scheduling algorithms.
  * The parcel subsystem: these changes improve the performance of the |hpx|
    networking layer, modularize the structure of the parcelports, and
    simplify the creation of new parcelports for other underlying networking
    libraries.
  * The API subsystem: these changes improved the conformance of the API to
    |cpp11|, extend and unify the available API functionality, and decrease
    the overheads created by various elements of the API.
  * The robustness of the component loading subsystem has been improved
    significantly, allowing to more portably and more reliably register the
    components needed by an application as startup. This additionally speeds up
    general application initialization.
* We added new API functionality like ``hpx::migrate`` and ``hpx::copy_component``
  which are the basic building blocks necessary for implementing higher level
  abstractions for system-wide load balancing, runtime-adaptive resource
  management, and object-oriented checkpointing and state-management.
* We removed the use of C++11 move emulation (using Boost.Move), replacing it
  with C++11 rvalue references. This is the first step towards using more and
  more native C++11 facilities which we plan to introduce in the future.
* We improved the reference counting scheme used by |hpx| which helps
  managing distributed objects and memory. This improves the overall stability
  of |hpx| and further simplifies writing real world applications.
* The minimal Boost version required to use |hpx| is now V1.49.0.
* This release coincides with the first release of |hpxpi| (V0.1.0), the
  first implementation of the |xpi_spec|_.

Bug fixes (closed tickets)
==========================

Here is a list of the important tickets we closed for this release.

* :hpx-issue:`1086` - Expose internal boost::shared_array to allow user
  management of array lifetime
* :hpx-issue:`1083` - Make shell examples copyable in docs
* :hpx-issue:`1080` - /threads{locality#*/total}/count/cumulative broken
* :hpx-issue:`1079` - Build problems on OS X
* :hpx-issue:`1078` - Improve robustness of component loading
* :hpx-issue:`1077` - Fix a missing enum definition for 'take' mode
* :hpx-issue:`1076` - Merge Jb master
* :hpx-issue:`1075` - Unknown CMake command "add_hpx_pseudo_target"
* :hpx-issue:`1074` - Implement ``apply_continue_callback`` and
  ``apply_colocated_callback``
* :hpx-issue:`1073` - The new ``apply_colocated`` and ``async_colocated``
  functions lead to automatic registered functions
* :hpx-issue:`1071` - Remove deferred_packaged_task
* :hpx-issue:`1069` - serialize_buffer with allocator fails at destruction
* :hpx-issue:`1068` - Coroutine include and forward declarations missing
* :hpx-issue:`1067` - Add allocator support to util::serialize_buffer
* :hpx-issue:`1066` - Allow for MPI_Init being called before HPX launches
* :hpx-issue:`1065` - AGAS cache isn't used/populated on worker localities
* :hpx-issue:`1064` - Reorder includes to ensure ws2 includes early
* :hpx-issue:`1063` - Add ``hpx::runtime::suspend`` and ``hpx::runtime::resume``
* :hpx-issue:`1062` - Fix ``async_continue`` to properly handle return types
* :hpx-issue:`1061` - Implement ``async_colocated`` and ``apply_colocated``
* :hpx-issue:`1060` - Implement minimal component migration
* :hpx-issue:`1058` - Remove ``HPX_UTIL_TUPLE`` from code base
* :hpx-issue:`1057` - Add performance counters for threading subsystem
* :hpx-issue:`1055` - Thread allocation uses two memory pools
* :hpx-issue:`1053` - Work stealing flawed
* :hpx-issue:`1052` - Fix a number of warnings
* :hpx-issue:`1049` - Fixes for TLS on OSX and more reliable test running
* :hpx-issue:`1048` - Fixing after 588 hang
* :hpx-issue:`1047` - Use port '0' for networking when using one locality
* :hpx-issue:`1046` - ``composable_guard`` test is broken when having more than
  one thread
* :hpx-issue:`1045` - Security missing headers
* :hpx-issue:`1044` - Native TLS on FreeBSD via __thread
* :hpx-issue:`1043` - ``async`` et.al. compute the wrong result type
* :hpx-issue:`1042` - ``async`` et.al. implicitly unwrap reference_wrappers
* :hpx-issue:`1041` - Remove redundant costly Kleene stars from regex searches
* :hpx-issue:`1040` - CMake script regex match patterns has unnecessary kleenes
* :hpx-issue:`1039` - Remove use of Boost.Move and replace with std::move and
  real rvalue refs
* :hpx-issue:`1038` - Bump minimal required Boost to 1.49.0
* :hpx-issue:`1037` - Implicit unwrapping of futures in async broken
* :hpx-issue:`1036` - Scheduler hangs when user code attempts to "block"
  OS-threads
* :hpx-issue:`1035` - Idle-rate counter always reports 100% idle rate
* :hpx-issue:`1034` - Symbolic name registration causes application hangs
* :hpx-issue:`1033` - Application options read in from an options file generate
  an error message
* :hpx-issue:`1032` - ``hpx::id_type`` local reference counting is wrong
* :hpx-issue:`1031` - Negative entry in reference count table
* :hpx-issue:`1030` - Implement condition_variable
* :hpx-issue:`1029` - Deadlock in thread scheduling subsystem
* :hpx-issue:`1028` - HPX-thread cumulative count performance counters report
  incorrect value
* :hpx-issue:`1027` - Expose ``hpx::thread_interrupted`` error code as a
  separate exception type
* :hpx-issue:`1026` - Exceptions thrown in asynchronous calls can be lost if the
  value of the future is never queried
* :hpx-issue:`1025` - ``future::wait_for``/``wait_until`` do not remove callback
* :hpx-issue:`1024` - Remove dependence to boost assert and create hpx assert
* :hpx-issue:`1023` - Segfaults with tcmalloc
* :hpx-issue:`1022` - prerequisites link in readme is broken
* :hpx-issue:`1020` - HPX Deadlock on external synchronization
* :hpx-issue:`1019` - Convert using ``BOOST_ASSERT`` to ``HPX_ASSERT``
* :hpx-issue:`1018` - compiling bug with gcc 4.8.1
* :hpx-issue:`1017` - Possible crash in io_pool executor
* :hpx-issue:`1016` - Crash at startup
* :hpx-issue:`1014` - Implement Increment/Decrement Merging
* :hpx-issue:`1013` - Add more logging channels to enable greater control over
  logging granularity
* :hpx-issue:`1012` - ``--hpx:debug-hpx-log`` and ``--hpx:debug-agas-log`` lead
  to non-thread safe writes
* :hpx-issue:`1011` - After installation, running applications from the
  build/staging directory no longer works
* :hpx-issue:`1010` - Mergeable decrement requests are not being merged
* :hpx-issue:`1009` - ``--hpx:list-symbolic-names`` crashes
* :hpx-issue:`1007` - Components are not properly destroyed
* :hpx-issue:`1006` - Segfault/hang in set_data
* :hpx-issue:`1003` - Performance counter naming issue
* :hpx-issue:`982` - Race condition during startup
* :hpx-issue:`912` - OS X: component type not found in map
* :hpx-issue:`663` - Create a buildbot slave based on Clang 3.2/OSX
* :hpx-issue:`636` - Expose ``this_locality::apply<act>(p1, p2);`` for local
  execution
* :hpx-issue:`197` - Add ``--console=address`` option for PBS runs
* :hpx-issue:`175` - Asynchronous AGAS API

