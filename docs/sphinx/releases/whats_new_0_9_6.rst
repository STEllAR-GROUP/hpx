..
    Copyright (C) 2007-2018 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_0_9_6:

===========================
|hpx| V0.9.6 (Jul 30, 2013)
===========================

We have had over 1200 commits since the last release and we have closed
roughly 140 tickets (bugs, feature requests, etc.).

General changes
===============

The major new features in this release are:

* We further consolidated the API exposed by |hpx|. We aligned our APIs as much
  as possible with the existing |cpp11|_ and related proposals to the C++
  standardization committee (such as |cpp11_n3632|_ and |cpp11_n3634|_).
* We implemented a first version of a distributed AGAS service which essentially
  eliminates all explicit AGAS network traffic.
* We created a native ibverbs parcelport allowing to take advantage of the
  superior latency and bandwidth characteristics of Infiniband networks.
* We successfully ported |hpx| to the Xeon Phi platform.
* Support for the SLURM scheduling system was implemented.
* Major efforts have been dedicated to improving the performance counter
  framework, numerous new counters were implemented and new APIs were added.
* We added a modular parcel compression system allowing to improve bandwidth
  utilization (by reducing the overall size of the transferred data).
* We added a modular parcel coalescing system allowing to combine several
  parcels into larger messages. This reduces latencies introduced by the
  communication layer.
* Added an experimental executors API allowing to use different scheduling
  policies for different parts of the code. This API has been modelled after the
  Standards proposal |cpp11_n3562|_. This API is bound to change in the future,
  though.
* Added minimal security support for localities which is enforced on the
  parcelport level. This support is preliminary and experimental and might
  change in the future.
* We created a parcelport using low level MPI functions. This is in support of
  legacy applications which are to be gradually ported and to support platforms
  where MPI is the only available portable networking layer.
* We added a preliminary and experimental implementation of a tuple-space object
  which exposes an interface similar to such systems described in the literature
  (see for instance |linda|_).

Bug fixes (closed tickets)
==========================

Here is a list of the important tickets we closed for this release. This is
again a very long list of newly implemented features and fixed issues.

* :hpx-issue:`806` - make (all) in examples folder does nothing
* :hpx-issue:`805` - Adding the introduction and fixing DOCBOOK dependencies for
  Windows use
* :hpx-issue:`804` - Add stackless (non-suspendable) thread type
* :hpx-issue:`803` - Create proper serialization support functions for
  util::tuple
* :hpx-issue:`800` - Add possibility to disable array optimizations during
  serialization
* :hpx-issue:`798` - HPX_LIMIT does not work for local dataflow
* :hpx-issue:`797` - Create a parcelport which uses MPI
* :hpx-issue:`796` - Problem with Large Numbers of Threads
* :hpx-issue:`793` - Changing dataflow test case to hang consistently
* :hpx-issue:`792` - CMake Error
* :hpx-issue:`791` - Problems with local::dataflow
* :hpx-issue:`790` - wait_for() doesn't compile
* :hpx-issue:`789` - HPX with Intel compiler segfaults
* :hpx-issue:`788` - Intel compiler support
* :hpx-issue:`787` - Fixed SFINAEd specializations
* :hpx-issue:`786` - Memory issues during benchmarking.
* :hpx-issue:`785` - Create an API allowing to register external threads with
  HPX
* :hpx-issue:`784` - util::plugin is throwing an error when a symbol is not
  found
* :hpx-issue:`783` - How does hpx:bind work?
* :hpx-issue:`782` - Added quotes around STRING REPLACE potentially empty
  arguments
* :hpx-issue:`781` - Make sure no exceptions propagate into the thread manager
* :hpx-issue:`780` - Allow arithmetics performance counters to expand its
  parameters
* :hpx-issue:`779` - Test case for 778
* :hpx-issue:`778` - Swapping futures segfaults
* :hpx-issue:`777` - hpx::lcos::details::when_xxx don't restore completion
  handlers
* :hpx-issue:`776` - Compiler chokes on dataflow overload with launch policy
* :hpx-issue:`775` - Runtime error with local dataflow (copying futures?)
* :hpx-issue:`774` - Using local dataflow without explicit namespace
* :hpx-issue:`773` - Local dataflow with unwrap: functor operators need to be
  const
* :hpx-issue:`772` - Allow (remote) actions to return a future
* :hpx-issue:`771` - Setting HPX_LIMIT gives huge boost MPL errors
* :hpx-issue:`770` - Add launch policy to (local) dataflow
* :hpx-issue:`769` - Make compile time configuration information available
* :hpx-issue:`768` - Const correctness problem in local dataflow
* :hpx-issue:`767` - Add launch policies to async
* :hpx-issue:`766` - Mark data structures for optimized (array based)
  serialization
* :hpx-issue:`765` - Align hpx::any with N3508: Any Library Proposal
  (Revision 2)
* :hpx-issue:`764` - Align hpx::future with newest N3558: A Standardized
  Representation of Asynchronous Operations
* :hpx-issue:`762` - added a human readable output for the ping pong example
* :hpx-issue:`761` - Ambiguous typename when constructing derived component
* :hpx-issue:`760` - Simple components can not be derived
* :hpx-issue:`759` - make install doesn't give a complete install
* :hpx-issue:`758` - Stack overflow when using locking_hook<>
* :hpx-issue:`757` - copy paste error; unsupported function overloading
* :hpx-issue:`756` - GTCX runtime issue in Gordon
* :hpx-issue:`755` - Papi counters don't work with reset and evaluate API's
* :hpx-issue:`753` - cmake bugfix and improved component action docs
* :hpx-issue:`752` - hpx simple component docs
* :hpx-issue:`750` - Add hpx::util::any
* :hpx-issue:`749` - Thread phase counter is not reset
* :hpx-issue:`748` - Memory performance counter are not registered
* :hpx-issue:`747` - Create performance counters exposing arithmetic operations
* :hpx-issue:`745` - apply_callback needs to invoke callback when applied
  locally
* :hpx-issue:`744` - CMake fixes
* :hpx-issue:`743` - Problem Building github version of HPX
* :hpx-issue:`742` - Remove HPX_STD_BIND
* :hpx-issue:`741` - assertion 'px != 0' failed: HPX(assertion_failure) for low
  numbers of OS threads
* :hpx-issue:`739` - Performance counters do not count to the end of the program
  or evaluation
* :hpx-issue:`738` - Dedicated AGAS server runs don't work; console ignores -a
  option.
* :hpx-issue:`737` - Missing bind overloads
* :hpx-issue:`736` - Performance counter wildcards do not always work
* :hpx-issue:`735` - Create native ibverbs parcelport based on rdma operations
* :hpx-issue:`734` - Threads stolen performance counter total is incorrect
* :hpx-issue:`733` - Test benchmarks need to be checked and fixed
* :hpx-issue:`732` - Build fails with Mac, using mac ports clang-3.3 on latest
  git branch
* :hpx-issue:`731` - Add global start/stop API for performance counters
* :hpx-issue:`730` - Performance counter values are apparently incorrect
* :hpx-issue:`729` - Unhandled switch
* :hpx-issue:`728` - Serialization of hpx::util::function between two localities
  causes seg faults
* :hpx-issue:`727` - Memory counters on Mac OS X
* :hpx-issue:`725` - Restore original thread priority on resume
* :hpx-issue:`724` - Performance benchmarks do not depend on main HPX libraries
* :hpx-issue:`723` - [teletype]--hpx:nodes=``cat $PBS_NODEFILE`` works;
  --hpx:nodefile=$PBS_NODEFILE does not.[c++]
* :hpx-issue:`722` - Fix binding const member functions as actions
* :hpx-issue:`719` - Create performance counter exposing compression ratio
* :hpx-issue:`718` - Add possibility to compress parcel data
* :hpx-issue:`717` - strip_credit_from_gid has misleading semantics
* :hpx-issue:`716` - Non-option arguments to programs run using ``pbsdsh`` must
  be before ``--hpx:nodes``, contrary to directions
* :hpx-issue:`715` - Re-thrown exceptions should retain the original call site
* :hpx-issue:`714` - failed assertion in debug mode
* :hpx-issue:`713` - Add performance counters monitoring connection caches
* :hpx-issue:`712` - Adjust parcel related performance counters to be connection
  type specific
* :hpx-issue:`711` - configuration failure
* :hpx-issue:`710` - Error "timed out while trying to find room in the
  connection cache" when trying to start multiple localities on a single
  computer
* :hpx-issue:`709` - Add new thread state 'staged' referring to task
  descriptions
* :hpx-issue:`708` - Detect/mitigate bad non-system installs of GCC on Redhat
  systems
* :hpx-issue:`707` - Many examples do not link with Git HEAD version
* :hpx-issue:`706` - ``hpx::init`` removes portions of non-option command line
  arguments before last ``=`` sign
* :hpx-issue:`705` - Create rolling average and median aggregating performance
  counters
* :hpx-issue:`704` - Create performance counter to expose thread queue waiting
  time
* :hpx-issue:`703` - Add support to HPX build system to find librcrtool.a and
  related headers
* :hpx-issue:`699` - Generalize instrumentation support
* :hpx-issue:`698` - compilation failure with hwloc absent
* :hpx-issue:`697` - Performance counter counts should be zero indexed
* :hpx-issue:`696` - Distributed problem
* :hpx-issue:`695` - Bad perf counter time printed
* :hpx-issue:`693` - ``--help`` doesn't print component specific command line
  options
* :hpx-issue:`692` - SLURM support broken
* :hpx-issue:`691` - exception while executing any application linked with hwloc
* :hpx-issue:`690` - thread_id_test and thread_launcher_test failing
* :hpx-issue:`689` - Make the buildbots use hwloc
* :hpx-issue:`687` - compilation error fix (hwloc_topology)
* :hpx-issue:`686` - Linker Error for Applications
* :hpx-issue:`684` - Pinning of service thread fails when number of worker
  threads equals the number of cores
* :hpx-issue:`682` - Add performance counters exposing number of stolen threads
* :hpx-issue:`681` - Add apply_continue for asynchronous chaining of actions
* :hpx-issue:`679` - Remove obsolete async_callback API functions
* :hpx-issue:`678` - Add new API for setting/triggering LCOs
* :hpx-issue:`677` - Add async_continue for true continuation style actions
* :hpx-issue:`676` - Buildbot for gcc 4.4 broken
* :hpx-issue:`675` - Partial preprocessing broken
* :hpx-issue:`674` - HPX segfaults when built with gcc 4.7
* :hpx-issue:`673` - ``use_guard_pages`` has inconsistent preprocessor guards
* :hpx-issue:`672` - External build breaks if library path has spaces
* :hpx-issue:`671` - release tarballs are tarbombs
* :hpx-issue:`670` - CMake won't find Boost headers in layout=versioned install
* :hpx-issue:`669` - Links in docs to source files broken if not installed
* :hpx-issue:`667` - Not reading ini file properly
* :hpx-issue:`664` - Adapt new meanings of 'const' and 'mutable'
* :hpx-issue:`661` - Implement BTL Parcel port
* :hpx-issue:`655` - Make HPX work with the "decltype" result_of
* :hpx-issue:`647` - documentation for specifying the number of high priority
  threads ``--hpx:high-priority-threads``
* :hpx-issue:`643` - Error parsing host file
* :hpx-issue:`642` - HWLoc issue with TAU
* :hpx-issue:`639` - Logging potentially suspends a running thread
* :hpx-issue:`634` - Improve error reporting from parcel layer
* :hpx-issue:`627` - Add tests for async and apply overloads that accept regular
  C++ functions
* :hpx-issue:`626` - hpx/future.hpp header
* :hpx-issue:`601` - Intel support
* :hpx-issue:`557` - Remove action codes
* :hpx-issue:`531` - AGAS request and response classes should use switch
  statements
* :hpx-issue:`529` - Investigate the state of hwloc support
* :hpx-issue:`526` - Make HPX aware of hyper-threading
* :hpx-issue:`518` - Create facilities allowing to use plain arrays as action
  arguments
* :hpx-issue:`473` - hwloc thread binding is broken on CPUs with hyperthreading
* :hpx-issue:`383` - Change result type detection for hpx::util::bind to use
  result_of protocol
* :hpx-issue:`341` - Consolidate route code
* :hpx-issue:`219` - Only copy arguments into actions once
* :hpx-issue:`177` - Implement distributed AGAS
* :hpx-issue:`43` - Support for Darwin (Xcode + Clang)

