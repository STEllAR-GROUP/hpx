..
    Copyright (C) 2007-2025 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_11_0:

============================
|hpx| V1.11.0 (Jun 30, 2025)
============================

This release is the last version of HPX that supports C++17. Future versions
of HPX will require compilation using C++20 or above.

General changes
===============

- Added synchronous versions of all collective operations. Added global predefined
  communicator objects that are accessible through new APIs:
  ``hpx::collectives::get_world_communicator()`` refers to all localities and
  ``hpx::collectives::get_local_communicator()`` refers to all threads on the
  calling locality. We unified the interfaces of the different communicator objects.
- Added ``hpx::experimental::run_on_all`` allowing to run a given function
  (possibly concurrently) using a given execution policy.
- Added a helper object ``hpx::runtime_manager`` that simplifies the initialization
  of HPX without needing to modify the ``main`` function of the application.
- We improved the compatibility with various accelerator frameworks (SYCL,
  OneAPI).
- Applied build system changes that allow building HPX without any prerequisites.
  This requires to pass ``-DHPX_WITH_FETCH_HWLOC=On`` and ``-DHPX_WITH_FETCH_BOOST=On``
  to the |cmake|_ configuration.
- We have performed a lot of code cleanup and refactoring to improve the overall
  code quality and decrease compile times.
- Added the ``hpx::contains`` and ``hpx::contains_subrange`` parallel algorithms.
- Adapted many of HPX' parallel algorithms to be usable with senders/receivers.

Breaking changes
================

- We have moved most of the APIs that were defined in the namespace
  ``hpx::parallel::execution`` to the namespace ``hpx::execution::experimental``.
  It was not possible to add compatibility facilities that will allow to continue
  using the old APIs, applications will have to be changed in order to
  continue functioning correctly.
- The CMake configuration parameter ``HPX_WITH_RUN_MAIN_EVERYWHERE`` is now
  deprecated and will be removed in the future. Use the preprocessor macro
  ``HPX_HAVE_RUN_MAIN_EVERYWHERE`` on a target-by-target case instead.
- Removed the dysfunctional libfabric parcelport.

- Removed features that were long deprecated (starting V1.8):
  - ``hpx::flush``, ``hpx::endl``, ``hpx::async_flush``, ``hpx::async_endl``
  - Various enumerator types are now only available as ``class enum`` requiring
  explicit scoped use of the enumerator values
  - Various non-conforming overloads of parallel algorithms
  - ``hpx::for_loop`` and friends (now only available as
    ``hpx::experimental::for_loop``)
  - ``hpx::parallel::induction`` is now only available as
    ``hpx::experimental::induction``
  - ``hpx::parallel::reduction`` and friends are now only available as
    ``hpx::experimental::reduction``
  - ``hpx::assertion::source_location`` is now only available as
    ``hpx::source_location``
  - ``hpx::lcos::split_future`` is now only available as ``hpx::split_future``
  - ``hpx::lcos::wait`` and friends have been removed altogether
  - ``hpx::lcos::wait_any`` and friends are now only available as
    ``hpx::wait_any``
  - ``hpx::lcos::wait_some`` and friends are now only available as
    ``hpx::wait_some``
  - ``hpx::lcos::wait_each`` and friends are now only available as
    ``hpx::wait_each``
  - ``hpx::lcos::wait_all`` and friends are now only available as
    ``hpx::wait_all``
  - ``hpx::lcos::when_all`` and friends are now only available as
    ``hpx::when_all``
  - ``hpx::lcos::when_any`` and friends are now only available as
    ``hpx::when_any``
  - ``hpx::lcos::when_each`` and friends are now only available as
    ``hpx::when_each``
  - ``hpx::lcos::when_some`` and friends are now only available as
    ``hpx::when_some``
  - ``hpx::util::optional`` and related facilities are now only available as
    ``hpx::optional``
  - ``hpx::util::bind`` and related facilities are now only available as
    ``hpx::bind``
  - ``hpx::util::function`` and friends are now only available as
    ``hpx::function``
  - ``hpx::traits::is_bound_action`` and related facilities are now
    only available as ``hpx::is_bound_action``
  - ``hpx::traits::is_bind_expression`` and related facilities are now
    only available as ``hpx::is_bind_expression``
  - ``hpx::traits::is_placeholder`` and related facilities are now
    only available as ``hpx::is_placeholder``
  - ``hpx::lcos::future`` and related facilities are now
    only available as ``hpx::future``
  - ``hpx::memory::intrusive_ptr`` is now only available as ``hpx::intrusive_ptr``
  - ``hpx::lcos::local::barrier`` is now only available as ``hpx::barrier``
  - ``hpx::lcos::barrier`` is now only available as ``hpx::distributed::barrier``
  - ``hpx::lcos::local::cpp20_binary_semaphore`` is now only available as
    ``hpx::detail::binary_semaphore``
  - ``hpx::lcos::local::condition_variable`` and friends are now only
    available as ``hpx::condition_variable``
  - ``hpx::lcos::local::counting_semaphore`` and friends are now only
    available as ``hpx::counting_semaphore``
  - ``hpx::lcos::local::cpp20_latch`` and is now only available as ``hpx::latch``
  - ``hpx::lcos::latch`` and is now only available as ``hpx::distributed::latch``
  - ``hpx::lcos::local::upgrade_lock`` and friends are now only available as
    ``hpx::upgrade_lock``
  - ``hpx::lcos::local::mutex`` and friends are now only available as
    ``hpx::mutex``
  - ``hpx::lcos::local::spinlock`` and friends are now only available as
    ``hpx::spinlock``
  - ``hpx::lcos::local::call_once`` and friends are now only available as
    ``hpx::call_once``
  - ``hpx::util::annotated_function`` and is now only available as
    ``hpx::annotated_function``
  - ``hpx::components::abstract_simple_component_base`` and is now only available as
    ``hpx::components::abstract_component_base``
  - ``hpx::naming::id_type`` and is now only available as ``hpx::id_type``

Closed issues
=============

* :hpx-issue:`6699` - Catch lower-level runtime error 
* :hpx-issue:`6696` - HPX master breaks with Kokkos
* :hpx-issue:`6691` - minimum_category doesn't work with custom iterator categories
* :hpx-issue:`6681` - build break - missing ';'
* :hpx-issue:`6658` - CMake error upon building HPX manually
* :hpx-issue:`6648` - Asio V1.34 deprecates io_context::work
* :hpx-issue:`6640` - iterator_facade doesn't work with custom iterator categories
* :hpx-issue:`6636` - problem with hpx::collectives::exclusive_scan
* :hpx-issue:`6623` - HPX serialization error with std::vector<std::vector<std::vector<float>>>
* :hpx-issue:`6616` - Add flux support to HPX to run on El Cap
* :hpx-issue:`6615` - Too many fails test after installed hpx
* :hpx-issue:`6605` - Partitionend vector copy constructor is broken
* :hpx-issue:`6586` - Bullet points in quick start/installing HPX section in documentation incorrectly rendered
* :hpx-issue:`6563` - Compilation issues on Grace Hopper
* :hpx-issue:`6544` - Errors in Public Distributed Api for all_to_all and gather_there
* :hpx-issue:`6519` - Option --hpx:queuing=local-priority-lifo is not configured
* :hpx-issue:`6501` - HPX 1.10 Failed Linking CXX executable for arm64-osx
* :hpx-issue:`5728` - Add optional fetch_content support for needed Boost libraries

Closed pull requests
====================

* :hpx-pr:`6716` - Fixing some of the reported linker warnings
* :hpx-pr:`6705` - Adding gcc/15 to jenkins
* :hpx-pr:`6701` - Attempting to fix shutdown hang on exception_info
* :hpx-pr:`6698` - Making sure .hpp.in files are not being installed
* :hpx-pr:`6697` - Minor docs fix
* :hpx-pr:`6695` - Adding missing ';'
* :hpx-pr:`6693` - Adding llvm/19 and 20 and cmake/4 Jenkins
* :hpx-pr:`6692` - Better implementation of minimal_category
* :hpx-pr:`6690` - Fixing bad #include in example
* :hpx-pr:`6689` - Fix unreachable code warning in wait_all
* :hpx-pr:`6687` - lci pp: change default ndevices=2 and progress_type=worker; improve document
* :hpx-pr:`6686` - lci pp: upgrade LCI autofetch target to 1.7.9
* :hpx-pr:`6685` - Improve run_on_all implementation and tests
* :hpx-pr:`6683` - Fix bad element comparison for reduce_by_key
* :hpx-pr:`6682` - Add C++23 std::generator equivalence test and fix missing semicolon
* :hpx-pr:`6680` - Add oneapi device init workaround
* :hpx-pr:`6679` - Fix sycl deprecations
* :hpx-pr:`6678` - Fix oneapi overloads
* :hpx-pr:`6677` - Offer a runtime manager object
* :hpx-pr:`6676` - Mention the HPX book
* :hpx-pr:`6675` - Bump required version of JSON library
* :hpx-pr:`6674` - Issue 6631
* :hpx-pr:`6673` - Fix: FindTBB.cmake cannot find correct TBB library. #6504
* :hpx-pr:`6672` - Update modules.rst
* :hpx-pr:`6670` - Add base template template param to execution_policy
* :hpx-pr:`6669` - Add execution policy support to run_on_all
* :hpx-pr:`6667` - Making sure bound threads are rescheduled on their original core
* :hpx-pr:`6666` - Improve documentation for reduction operations
* :hpx-pr:`6664` - Fix CMake template when fetching Boost
* :hpx-pr:`6663` - More run_on_all overloads
* :hpx-pr:`6662` - Fix "unary minus operator applied to unsigned type" warning
* :hpx-pr:`6661` - Adding simple experimental::run_on_all
* :hpx-pr:`6659` - fix(reduce): Initialize accumulator with init instead of first element
* :hpx-pr:`6656` - Add missing channel_communicator::get_info
* :hpx-pr:`6652` - Adding channel-based ping-pong example
* :hpx-pr:`6650` - Adding constructor overloads to partitioned_vector
* :hpx-pr:`6649` - Remove the use of deprecated asio::io_context::work
* :hpx-pr:`6645` - Fixing collectives::exclusive_scan
* :hpx-pr:`6644` - Update result_type in set_union.hpp
* :hpx-pr:`6643` - Update result_type in set_union.hpp
* :hpx-pr:`6642` - Allowing to use custom iterator tags with iterator_facade
* :hpx-pr:`6641` - Allowing for zip-iterator to refer to sequences of different length
* :hpx-pr:`6639` - docs: Fix spelling in example dictionary
* :hpx-pr:`6638` - Update set_union.hpp
* :hpx-pr:`6637` - lci/mpi pp: fix the case when non-zero-copy data is larger than INT_MAX
* :hpx-pr:`6635` - Adding simplified reduction overload
* :hpx-pr:`6634` - Fixed issue 6634: Unqualified calls to insertion_sort
* :hpx-pr:`6633` - Increase timeouts for CircleCI tests
* :hpx-pr:`6630` - Fix CPUId test
* :hpx-pr:`6628` - Link aclocal with aclocal-1.16 as hwloc asks for it
* :hpx-pr:`6626` - Fixing MPI parcel port issue exposed by #6623
* :hpx-pr:`6622` - Newbranch:HPX-Based Task Scheduler with CUDA-Quantum Integration & Benchmarking
* :hpx-pr:`6621` - HPX-Based Task Scheduler with CUDA-Quantum Integration & Benchmarking
* :hpx-pr:`6620` - new test: very big tchunk
* :hpx-pr:`6619` - mpi pp: fix transmission chunk send
* :hpx-pr:`6617` - Adding support for the Flux job scheduling environment
* :hpx-pr:`6614` - Fix fallback to module mode for CMake finding Boost
* :hpx-pr:`6613` - Fix partitioned_vector_handle_values test
* :hpx-pr:`6612` - Fixing naming convention for pp constant
* :hpx-pr:`6611` - Fix Hwloc fetch content
* :hpx-pr:`6610` - Add docs for synchronous collective operations
* :hpx-pr:`6609` - Update perftest CI reference measurements
* :hpx-pr:`6608` - Partially support data parallel for_loop
* :hpx-pr:`6607` - Cleaning up copy_component facility
* :hpx-pr:`6606` - Making sure copy_component creates a new gid
* :hpx-pr:`6600` - Fixing sync collectives
* :hpx-pr:`6599` - Make HPX_HAVE_RUN_MAIN_EVERYWHERE application specific
* :hpx-pr:`6598` - Adding synchronous collective operations
* :hpx-pr:`6596` - Minor fixes and optimizations
* :hpx-pr:`6595` - Rfa parallel
* :hpx-pr:`6594` - Move get_stack_ptr to source
* :hpx-pr:`6593` - Fix outdated documentation and missing flags
* :hpx-pr:`6592` - HPX_HAVE_THREADS_GET_STACK_POINTER to match builtin_frame_address feature test
* :hpx-pr:`6591` - Feature test for __builtin_frame_address
* :hpx-pr:`6590` - Add device guard for noexcept
* :hpx-pr:`6587` - Fix bullet points in Quickstart
* :hpx-pr:`6585` - Fixed escape characters format to handle warning due to misinterpretation of syntax
* :hpx-pr:`6583` - Execute feature test for at_quick_exit
* :hpx-pr:`6582` - Accommodate for CircleCI reduce available number of cores to two
* :hpx-pr:`6581` - Attempting to work around a Boost.Spirit problem
* :hpx-pr:`6580` - mpi pp: fix messages larger than INT_MAX
* :hpx-pr:`6578` - Remove leftovers from libfabric parcelport
* :hpx-pr:`6577` - Download Boost from their own archives, not from Sourceforge
* :hpx-pr:`6576` - Fix CMake warning issued since CMake V3.30
* :hpx-pr:`6575` - Replace previously downloaded CDash conv.xsl with local version
* :hpx-pr:`6570` - Update exception_list.hpp
* :hpx-pr:`6569` - Update exception_list.hpp
* :hpx-pr:`6567` - Fix vectorization error on copy algorithm
* :hpx-pr:`6566` - lci pp: fix messages larger than INT_MAX
* :hpx-pr:`6565` - Moving most of APIs from hpx::parallel::execution to hpx::execution::experimental
* :hpx-pr:`6564` - Remove superfluous HPX_MOVE()
* :hpx-pr:`6562` - Fix doc return type of broadcast_to
* :hpx-pr:`6560` - Fixes for bit_cast on 32bit systems
* :hpx-pr:`6559` - Making sure that all parcelport counters are unavailable if no networking is needed or configured
* :hpx-pr:`6558` - Remove CSCS CI's
* :hpx-pr:`6556` - Set copyright year in generated files
* :hpx-pr:`6553` - Fix omp vectorization pragma errors
* :hpx-pr:`6551` - Update building_hpx.rst
* :hpx-pr:`6550` - Partitioned vector updates
* :hpx-pr:`6549` - Fix CMake conditionals checking ENV variables
* :hpx-pr:`6548` - Update CONTRIBUTING.md
* :hpx-pr:`6546` - Fix incorrect signature of distributed API functions
* :hpx-pr:`6543` - Throwing an exception derived from std::bad_alloc on OOM conditions
* :hpx-pr:`6539` - Use thread-safe cache in thread_local_caching_allocator
* :hpx-pr:`6537` - Update README.rst
* :hpx-pr:`6531` - More fixes for the Boost package
* :hpx-pr:`6527` - Improve the LCI parcelport documentation
* :hpx-pr:`6525` - Addressing cmake warnings issued starting V3.30
* :hpx-pr:`6522` - Fixing distance test
* :hpx-pr:`6520` - Adding optional handshakes to acknowledge the received data
* :hpx-pr:`6518` - Make sure that --hpx:ini log settings take effect
* :hpx-pr:`6512` - Minor cleanup of future_data
* :hpx-pr:`6510` - Include Boost as CMake subproject
* :hpx-pr:`6509` - Add components documentation
* :hpx-pr:`6508` - Fix typo: s/unititiallized/uninitialized/
* :hpx-pr:`6507` - Update LSU Jenkins libraries to match Rostam 3.0 with RHEL9
* :hpx-pr:`6503` - Fix 2 tests on FreeBSD by initializing freebsd_environ
* :hpx-pr:`6499` - Fix crash in get_executable_filename on FreeBSD
* :hpx-pr:`6498` - Avoid rewriting defines.hpp
* :hpx-pr:`6497` - Contains and contains_subrange parallel algorithm implementation GSOC 2024
* :hpx-pr:`6496` - Prevent usage of CMake try_run on crosscompiling
* :hpx-pr:`6494` - Add unit test cases and fixes for the S/R versions of the parallel algorithms
* :hpx-pr:`6487` - Fixing security vulnerabilities reported by MSVC security checks
* :hpx-pr:`6486` - Create codeql.yml
* :hpx-pr:`6474` - Remove remnants of libfabric parcelport
* :hpx-pr:`6473` - Add documentation for distributed implementations of post, async, sync and dataflow
* :hpx-pr:`6471` - Add distance.cpp test in CMake
* :hpx-pr:`6468` - Small vector relocation
* :hpx-pr:`6448` - Standardising Benchmarks, with support for nanobench as an option for its backend
* :hpx-pr:`6365` - Release V1.10.0
* :hpx-pr:`6089` - Implementing p2079


