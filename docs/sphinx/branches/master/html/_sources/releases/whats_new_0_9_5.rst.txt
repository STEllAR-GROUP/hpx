..
    Copyright (C) 2007-2018 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_0_9_5:

===========================
|hpx| V0.9.5 (Jan 16, 2013)
===========================

We have had over 1000 commits since the last release and we have closed roughly
150 tickets (bugs, feature requests, etc.).

General changes
===============

This release is continuing along the lines of code and API consolidation, and
overall usability inprovements. We dedicated much attention to performance and
we were able to significantly improve the threading and networking subsystems.

We successfully ported |hpx| to the Android platform. |hpx| applications now not
only can run on mobile devices, but we support heterogeneous applications
running across architecture boundaries. At the Supercomputing Conference 2012 we
demonstrated connecting Android tablets to simulations running on a Linux
cluster. The Android tablet was used to query performance counters from the
Linux simulation and to steer its parameters.

We successfully ported |hpx| to Mac OSX (using the Clang compiler). Thanks to
Pyry Jahkola for contributing the corresponding patches. Please see the section
:ref:`macos_installation` for more details.

We made a special effort to make HPX usable in highly concurrent use cases. Many
of the HPX API functions which possibly take longer than 100 microseconds to
execute now can be invoked asynchronously. We added uniform support for
composing futures which simplifies to write asynchronous code. HPX actions
(function objects encapsulating possibly concurrent remote function invocations)
are now well integrated with all other API facilities such like ``hpx::bind``.

All of the API has been aligned as much as possible with established paradigms.
HPX now mirrors many of the facilities as defined in the |cpp11|, such as
``hpx::thread``, ``hpx::function``, ``hpx::future``, etc.

A lot of work has been put into improving the documentation. Many of the API
functions are documented now, concepts are explained in detail, and examples are
better described than before. The new documentation index enables finding
information with lesser effort.

This is the first release of HPX we perform after the move to |hpx_github|_ This
step has enabled a wider participation from the community and further encourages
us in our decision to release HPX as a true open source library (HPX is licensed
under the very liberal |boost_license|_).

Bug fixes (closed tickets)
==========================

Here is a list of the important tickets we closed for this release. This is by
far the longest list of newly implemented features and fixed issues for any of
HPX' releases so far.

* :hpx-issue:`666` - Segfault on calling hpx::finalize twice
* :hpx-issue:`665` - Adding declaration num_of_cores
* :hpx-issue:`662` - pkgconfig is building wrong
* :hpx-issue:`660` - Need uninterrupt function
* :hpx-issue:`659` - Move our logging library into a different namespace
* :hpx-issue:`658` - Dynamic performance counter types are broken
* :hpx-issue:`657` - HPX v0.9.5 (RC1) hello_world example segfaulting
* :hpx-issue:`656` - Define the affinity of parcel-pool, io-pool, and timer-pool
  threads
* :hpx-issue:`654` - Integrate the Boost auto_index tool with documentation
* :hpx-issue:`653` - Make HPX build on OS X + Clang + libc++
* :hpx-issue:`651` - Add fine-grained control for thread pinning
* :hpx-issue:`650` - Command line no error message when using -hpx:(anything)
* :hpx-issue:`645` - Command line aliases don't work in [teletype]``@file``[c++]
* :hpx-issue:`644` - Terminated threads are not always properly cleaned up
* :hpx-issue:`640` - ``future_data<T>::set_on_completed_`` used without locks
* :hpx-issue:`638` - hpx build with intel compilers fails on linux
* :hpx-issue:`637` - --copy-dt-needed-entries breaks with gold
* :hpx-issue:`635` - Boost V1.53 will add Boost.Lockfree and Boost.Atomic
* :hpx-issue:`633` - Re-add examples to final 0.9.5 release
* :hpx-issue:`632` - Example ``thread_aware_timer`` is broken
* :hpx-issue:`631` - FFT application throws error in parcellayer
* :hpx-issue:`630` - Event synchronization example is broken
* :hpx-issue:`629` - Waiting on futures hangs
* :hpx-issue:`628` - Add an ``HPX_ALWAYS_ASSERT`` macro
* :hpx-issue:`625` - Port coroutines context switch benchmark
* :hpx-issue:`621` - New INI section for stack sizes
* :hpx-issue:`618` - pkg_config support does not work with a HPX debug build
* :hpx-issue:`617` -
  hpx/external/logging/boost/logging/detail/cache_before_init.hpp:139:67: error:
  'get_thread_id' was not declared in this scope
* :hpx-issue:`616` - Change wait_xxx not to use locking
* :hpx-issue:`615` - Revert visibility 'fix'
  (fb0b6b8245dad1127b0c25ebafd9386b3945cca9)
* :hpx-issue:`614` - Fix Dataflow linker error
* :hpx-issue:`613` - find_here should throw an exception on failure
* :hpx-issue:`612` - Thread phase doesn't show up in debug mode
* :hpx-issue:`611` - Make stack guard pages configurable at runtime
  (initialization time)
* :hpx-issue:`610` - Co-Locate Components
* :hpx-issue:`609` - future_overhead
* :hpx-issue:`608` - ``--hpx:list-counter-infos`` problem
* :hpx-issue:`607` - Update Boost.Context based backend for coroutines
* :hpx-issue:`606` - 1d_wave_equation is not working
* :hpx-issue:`605` - Any C++ function that has serializable arguments and a
  serializable return type should be remotable
* :hpx-issue:`604` - Connecting localities isn't working anymore
* :hpx-issue:`603` - Do not verify any ini entries read from a file
* :hpx-issue:`602` - Rename argument_size to type_size/ added implementation to
  get parcel size
* :hpx-issue:`599` - Enable locality specific command line options
* :hpx-issue:`598` - Need an API that accesses the performance counter reporting
  the system uptime
* :hpx-issue:`597` - compiling on ranger
* :hpx-issue:`595` - I need a place to store data in a thread self pointer
* :hpx-issue:`594` - 32/64 interoperability
* :hpx-issue:`593` - Warn if logging is disabled at compile time but requested
  at runtime
* :hpx-issue:`592` - Add optional argument value to ``--hpx:list-counters`` and
  ``--hpx:list-counter-infos``
* :hpx-issue:`591` - Allow for wildcards in performance counter names specified
  with ``--hpx:print-counter``
* :hpx-issue:`590` - Local promise semantic differences
* :hpx-issue:`589` - Create API to query performance counter names
* :hpx-issue:`587` - Add get_num_localities and get_num_threads to AGAS API
* :hpx-issue:`586` - Adjust local AGAS cache size based on number of localities
* :hpx-issue:`585` - Error while using counters in HPX
* :hpx-issue:`584` - counting argument size of actions, initial pass.
* :hpx-issue:`581` - Remove ``RemoteResult`` template parameter for ``future<>``
* :hpx-issue:`580` - Add possibility to hook into actions
* :hpx-issue:`578` - Use angle brackets in HPX error dumps
* :hpx-issue:`576` - Exception incorrectly thrown when ``--help`` is used
* :hpx-issue:`575` - HPX(bad_component_type) with gcc 4.7.2 and boost 1.51
* :hpx-issue:`574` - ``--hpx:connect`` command line parameter not working
  correctly
* :hpx-issue:`571` - ``hpx::wait()`` (callback version) should pass the future
  to the callback function
* :hpx-issue:`570` - ``hpx::wait`` should operate on ``boost::arrays`` and
  ``std::lists``
* :hpx-issue:`569` - Add a logging sink for Android
* :hpx-issue:`568` - 2-argument version of ``HPX_DEFINE_COMPONENT_ACTION``
* :hpx-issue:`567` - Connecting to a running HPX application works only once
* :hpx-issue:`565` - HPX doesn't shutdown properly
* :hpx-issue:`564` - Partial preprocessing of new component creation interface
* :hpx-issue:`563` - Add ``hpx::start``/``hpx::stop`` to avoid blocking main
  thread
* :hpx-issue:`562` - All command line arguments swallowed by hpx
* :hpx-issue:`561` - Boost.Tuple is not move aware
* :hpx-issue:`558` - ``boost::shared_ptr<>`` style semantics/syntax for client
  classes
* :hpx-issue:`556` - Creation of partially preprocessed headers should be
  enabled for Boost newer than V1.50
* :hpx-issue:`555` - ``BOOST_FORCEINLINE`` does not name a type
* :hpx-issue:`554` - Possible race condition in thread ``get_id()``
* :hpx-issue:`552` - Move enable client_base
* :hpx-issue:`550` - Add stack size category 'huge'
* :hpx-issue:`549` - ShenEOS run seg-faults on single or distributed runs
* :hpx-issue:`545` - ``AUTOGLOB`` broken for add_hpx_component
* :hpx-issue:`542` - FindHPX_HDF5 still searches multiple times
* :hpx-issue:`541` - Quotes around application name in hpx::init
* :hpx-issue:`539` - Race conditition occurring with new lightweight threads
* :hpx-issue:`535` - hpx_run_tests.py exits with no error code when tests are
  missing
* :hpx-issue:`530` - Thread description(<unknown>) in logs
* :hpx-issue:`523` - Make thread objects more lightweight
* :hpx-issue:`521` - ``hpx::error_code`` is not usable for lightweight error
  handling
* :hpx-issue:`520` - Add full user environment to HPX logs
* :hpx-issue:`519` - Build succeeds, running fails
* :hpx-issue:`517` - Add a guard page to linux coroutine stacks
* :hpx-issue:`516` - hpx::thread::detach suspends while holding locks, leads to
  hang in debug
* :hpx-issue:`514` - Preprocessed headers for <hpx/apply.hpp> don't compile
* :hpx-issue:`513` - Buildbot configuration problem
* :hpx-issue:`512` - Implement action based stack size customization
* :hpx-issue:`511` - Move action priority into a separate type trait
* :hpx-issue:`510` - trunk broken
* :hpx-issue:`507` - no matching function for call to
  ``boost::scoped_ptr<hpx::threads::topology>::scoped_ptr(hpx::threads::linux_topology*)``
* :hpx-issue:`505` - undefined_symbol regression test currently failing
* :hpx-issue:`502` - Adding OpenCL and OCLM support to HPX for Windows and Linux
* :hpx-issue:`501` - find_package(HPX) sets cmake output variables
* :hpx-issue:`500` - wait_any/wait_all are badly named
* :hpx-issue:`499` - Add support for disabling pbs support in pbs runs
* :hpx-issue:`498` - Error during no-cache runs
* :hpx-issue:`496` - Add partial preprocessing support to cmake
* :hpx-issue:`495` - Support HPX modules exporting startup/shutdown functions
  only
* :hpx-issue:`494` - Allow modules to specify when to run startup/shutdown
  functions
* :hpx-issue:`493` - Avoid constructing a string in make_success_code
* :hpx-issue:`492` - Performance counter creation is no longer synchronized at
  startup
* :hpx-issue:`491` - Performance counter creation is no longer synchronized at
  startup
* :hpx-issue:`490` - Sheneos on_completed_bulk seg fault in distributed
* :hpx-issue:`489` - compiling issue with g++44
* :hpx-issue:`488` - Adding OpenCL and OCLM support to HPX for the MSVC platform
* :hpx-issue:`487` - FindHPX.cmake problems
* :hpx-issue:`485` - Change distributing_factory and binpacking_factory to use
  bulk creation
* :hpx-issue:`484` - Change ``HPX_DONT_USE_PREPROCESSED_FILES`` to
  ``HPX_USE_PREPROCESSED_FILES``
* :hpx-issue:`483` - Memory counter for Windows
* :hpx-issue:`479` - strange errors appear when requesting performance counters
  on multiple nodes
* :hpx-issue:`477` - Create (global) timer for multi-threaded measurements
* :hpx-issue:`472` - Add partial preprocessing using Wave
* :hpx-issue:`471` - Segfault stack traces don't show up in release
* :hpx-issue:`468` - External projects need to link with internal components
* :hpx-issue:`462` - Startup/shutdown functions are called more than once
* :hpx-issue:`458` - Consolidate hpx::util::high_resolution_timer and
  ``hpx::util::high_resolution_clock``
* :hpx-issue:`457` - index out of bounds in ``allgather_and_gate`` on 4 cores or
  more
* :hpx-issue:`448` - Make HPX compile with clang
* :hpx-issue:`447` - 'make tests' should execute tests on local installation
* :hpx-issue:`446` - Remove SVN-related code from the codebase
* :hpx-issue:`444` - race condition in smp
* :hpx-issue:`441` - Patched Boost.Serialization headers should only be
  installed if needed
* :hpx-issue:`439` - Components using ``HPX_REGISTER_STARTUP_MODULE`` fail to
  compile with MSVC
* :hpx-issue:`436` - Verify that no locks are being held while threads are
  suspended
* :hpx-issue:`435` - Installing HPX should not clobber existing Boost
  installation
* :hpx-issue:`434` - Logging external component failed (Boost 1.50)
* :hpx-issue:`433` - Runtime crash when building all examples
* :hpx-issue:`432` - Dataflow hangs on 512 cores/64 nodes
* :hpx-issue:`430` - Problem with distributing factory
* :hpx-issue:`424` - File paths referring to XSL-files need to be properly
  escaped
* :hpx-issue:`417` - Make dataflow LCOs work out of the box by using partial
  preprocessing
* :hpx-issue:`413` - hpx_svnversion.py fails on Windows
* :hpx-issue:`412` - Make hpx::error_code equivalent to hpx::exception
* :hpx-issue:`398` - HPX clobbers out-of-tree application specific CMake
  variables (specifically ``CMAKE_BUILD_TYPE``)
* :hpx-issue:`394` - Remove code generating random port numbers for network
* :hpx-issue:`378` - ShenEOS scaling issues
* :hpx-issue:`354` - Create a coroutines wrapper for Boost.Context
* :hpx-issue:`349` - Commandline option ``--localities=N/-lN`` should be
  necessary only on AGAS locality
* :hpx-issue:`334` - Add auto_index support to cmake based documentation
  toolchain
* :hpx-issue:`318` - Network benchmarks
* :hpx-issue:`317` - Implement network performance counters
* :hpx-issue:`310` - Duplicate logging entries
* :hpx-issue:`230` - Add compile time option to disable thread debugging info
* :hpx-issue:`171` - Add an INI option to turn off deadlock detection
  independently of logging
* :hpx-issue:`170` - OSHL internal counters are incorrect
* :hpx-issue:`103` - Better diagnostics for multiple component/action
  registerations under the same name
* :hpx-issue:`48` - Support for Darwin (Xcode + Clang)
* :hpx-issue:`21` - Build fails with GCC 4.6

