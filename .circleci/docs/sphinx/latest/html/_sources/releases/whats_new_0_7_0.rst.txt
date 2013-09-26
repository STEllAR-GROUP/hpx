..
    Copyright (C) 2007-2018 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_0_7_0:

===========================
|hpx| V0.7.0 (Dec 12, 2011)
===========================

We have had roughly 1000 commits since the last release and we have closed
approximately 120 tickets (bugs, feature requests, etc.).

General changes
===============

* Completely removed code related to deprecated AGAS V1, started to work on AGAS
  V2.1.
* Started to clean up and streamline the exposed APIs (see 'API changes' below
  for more details).
* Revamped and unified performance counter framework, added a lot of new
  performance counter instances for monitoring of a diverse set of internal
  |hpx| parameters (queue lengths, access statistics, etc.).
* Improved general error handling and logging support.
* Fixed several race conditions, improved overall stability, decreased memory
  footprint, improved overall performance (major optimizations include native
  TLS support and ranged-based AGAS caching).
* Added support for running |hpx| applications with PBS.
* Many updates to the build system, added support for gcc 4.5.x and 4.6.x, added
  C++11 support.
* Many updates to default command line options.
* Added many tests, set up buildbot for continuous integration testing.
* Better shutdown handling of distributed applications.

Example applications
====================

* quickstart/factorial and quickstart/fibonacci, future-recursive parallel
  algorithms.
* quickstart/hello_world, distributed hello world example.
* quickstart/rma, simple remote memory access example
* quickstart/quicksort, parallel quicksort implementation.
* gtc, gyrokinetic torodial code.
* bfs, breadth-first-search, example code for a graph application.
* sheneos, partitioning of large data sets.
* accumulator, simple component example.
* balancing/os_thread_num, balancing/px_thread_phase, examples demonstrating
  load balancing and work stealing.

API changes
===========

* Added ``hpx::find_all_localities``.
* Added ``hpx::terminate`` for non-graceful termination of applications.
* Added ``hpx::lcos::async`` functions for simpler asynchronous programming.
* Added new AGAS interface for handling of symbolic namespace
  (``hpx::agas::*``).
* Renamed ``hpx::components::wait`` to ``hpx::lcos::wait``.
* Renamed ``hpx::lcos::future_value`` to ``hpx::lcos::promise``.
* Renamed ``hpx::lcos::recursive_mutex`` to
  ``hpx::lcos::local_recursive_mutex``, ``hpx::lcos::mutex`` to
  ``hpx::lcos::local_mutex``
* Removed support for Boost versions older than V1.38, recommended Boost version
  is now V1.47 and newer.
* Removed ``hpx::process`` (this will be replaced by a real process
  implementation in the future).
* Removed non-functional LCO code (``hpx::lcos::dataflow``,
  ``hpx::lcos::thunk``, ``hpx::lcos::dataflow_variable``).
* Removed deprecated ``hpx::naming::full_address``.

Bug fixes (closed tickets)
==========================

Here is a list of the important tickets we closed for this release:

* :hpx-issue:`28` - Integrate Windows/Linux CMake code for |hpx| core
* :hpx-issue:`32` - hpx::cout() should be hpx::cout
* :hpx-issue:`33` - AGAS V2 legacy client does not properly handle error_code
* :hpx-issue:`60` - AGAS: allow for registerid to optionally take ownership of
  the gid
* :hpx-issue:`62` - adaptive1d compilation failure in Fusion
* :hpx-issue:`64` - Parcel subsystem doesn't resolve domain names
* :hpx-issue:`83` - No error handling if no console is available
* :hpx-issue:`84` - No error handling if a hosted locality is treated as the
  bootstrap server
* :hpx-issue:`90` - Add general commandline option -N
* :hpx-issue:`91` - Add possibility to read command line arguments from file
* :hpx-issue:`92` - Always log exceptions/errors to the log file
* :hpx-issue:`93` - Log the command line/program name
* :hpx-issue:`95` - Support for distributed launches
* :hpx-issue:`97` - Attempt to create a bad component type in AMR examples
* :hpx-issue:`100` - factorial and factorial_get examples trigger AGAS component
  type assertions
* :hpx-issue:`101` - Segfault when hpx::process::here() is called in fibonacci2
* :hpx-issue:`102` - unknown_component_address in int_object_semaphore_client
* :hpx-issue:`114` - marduk raises assertion with default parameters
* :hpx-issue:`115` - Logging messages for SMP runs (on the console) shouldn't be
  buffered
* :hpx-issue:`119` - marduk linking strategy breaks other applications
* :hpx-issue:`121` - pbsdsh problem
* :hpx-issue:`123` - marduk, dataflow and adaptive1d fail to build
* :hpx-issue:`124` - Lower default preprocessing arity
* :hpx-issue:`125` - Move hpx::detail::diagnostic_information out of the detail
  namespace
* :hpx-issue:`126` - Test definitions for AGAS reference counting
* :hpx-issue:`128` - Add averaging performance counter
* :hpx-issue:`129` - Error with endian.hpp while building adaptive1d
* :hpx-issue:`130` - Bad initialization of performance counters
* :hpx-issue:`131` - Add global startup/shutdown functions to component modules
* :hpx-issue:`132` - Avoid using auto_ptr
* :hpx-issue:`133` - On Windows hpx.dll doesn't get installed
* :hpx-issue:`134` - HPX_LIBRARY does not reflect real library name (on Windows)
* :hpx-issue:`135` - Add detection of unique_ptr to build system
* :hpx-issue:`137` - Add command line option allowing to repeatedly evaluate
  performance counters
* :hpx-issue:`139` - Logging is broken
* :hpx-issue:`140` - CMake problem on windows
* :hpx-issue:`141` - Move all non-component libraries into $PREFIX/lib/hpx
* :hpx-issue:`143` - adaptive1d throws an exception with the default command
  line options
* :hpx-issue:`146` - Early exception handling is broken
* :hpx-issue:`147` - Sheneos doesn't link on Linux
* :hpx-issue:`149` - sheneos_test hangs
* :hpx-issue:`154` - Compilation fails for r5661
* :hpx-issue:`155` - Sine performance counters example chokes on chrono headers
* :hpx-issue:`156` - Add build type to --version
* :hpx-issue:`157` - Extend AGAS caching to store gid ranges
* :hpx-issue:`158` - r5691 doesn't compile
* :hpx-issue:`160` - Re-add AGAS function for resolving a locality to its prefix
* :hpx-issue:`168` - Managed components should be able to access their own GID
* :hpx-issue:`169` - Rewrite AGAS future pool
* :hpx-issue:`179` - Complete switch to request class for AGAS server interface
* :hpx-issue:`182` - Sine performance counter is loaded by other examples
* :hpx-issue:`185` - Write tests for symbol namespace reference counting
* :hpx-issue:`191` - Assignment of read-only variable in point_geometry
* :hpx-issue:`200` - Seg faults when querying performance counters
* :hpx-issue:`204` - --ifnames and suffix stripping needs to be more generic
* :hpx-issue:`205` - --list-* and --print-counter-* options do not work together
  and produce no warning
* :hpx-issue:`207` - Implement decrement entry merging
* :hpx-issue:`208` - Replace the spinlocks in AGAS with hpx::lcos::local_mutexes
* :hpx-issue:`210` - Add an --ifprefix option
* :hpx-issue:`214` - Performance test for PX-thread creation
* :hpx-issue:`216` - VS2010 compilation
* :hpx-issue:`222` - r6045 context_linux_x86.hpp
* :hpx-issue:`223` - fibonacci hangs when changing the state of an active thread
* :hpx-issue:`225` - Active threads end up in the FEB wait queue
* :hpx-issue:`226` - VS Build Error for Accumulator Client
* :hpx-issue:`228` - Move all traits into namespace hpx::traits
* :hpx-issue:`229` - Invalid initialization of reference in thread_init_data
* :hpx-issue:`235` - Invalid GID in iostreams
* :hpx-issue:`238` - Demangle type names for the default implementation of
  get_action_name
* :hpx-issue:`241` - C++11 support breaks GCC 4.5
* :hpx-issue:`247` - Reference to temporary with GCC 4.4
* :hpx-issue:`248` - Seg fault at shutdown with GCC 4.4
* :hpx-issue:`253` - Default component action registration kills compiler
* :hpx-issue:`272` - G++ unrecognized command line option
* :hpx-issue:`273` - quicksort example doesn't compile
* :hpx-issue:`277` - Invalid CMake logic for Windows

