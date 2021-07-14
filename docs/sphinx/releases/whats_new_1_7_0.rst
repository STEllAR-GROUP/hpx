..
    Copyright (C) 2020-2021 ETH Zurich
    Copyright (C) 2007-2020 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_7_0:

===========================
|hpx| V1.7.0 (Jul 14, 2021)
===========================

This release is again focused on C++20 conformance of algorithms. Additionally,
many new experimental sender-based algorithms have been added based on the
latest proposals.

General changes
===============

- The following algorithms have been adapted to be C++20 conformant:

  - ``remove``,
  - ``remove_if``,
  - ``remove_copy``,
  - ``remove_copy_if``,
  - ``replace``,
  - ``replace_if``,
  - ``reverse``, and
  - ``lexicographical_compare``.

- When the compiler and standard library support the standard execution policies
  ``std::execution::seq``, ``std::execution::par``, and
  ``std::execution::par_unseq`` they can now be used in all |hpx| parallel
  algorithms with equivalent behaviour to the non-task policies
  ``hpx::execution::seq``, ``hpx::execution::par``, and
  ``hpx::execution::par_unseq``.
- Vc support has been fixed, after being broken in 1.6.0. In addition, |hpx| now
  experimentally supports GCC's SIMD implementation, when available. The
  implementation can be used through the ``hpx::execution::simd`` and
  ``hpx::execution::simdpar`` execution policies.
- The customization points ``sync_execute``, ``async_execute``,
  ``then_execute``, ``post``, ``bulk_sync_execute``, ``bulk_async_execute``, and
  ``bulk_then_execute`` are now implemented using ``tag_dispatch`` (previously
  ``tag_invoke``). Executors can still be implemented by providing the
  aforementioned functions as member functions of an executor.
- New functionality, enhancements, and fixes based on P0443r14 (executors
  proposal) and P1897 (sender-based algorithms) have been added to the
  ``hpx::execution::experimental`` namespace. These can be accessed through the
  ``hpx/execution.hpp`` and ``hpx/local/execution.hpp`` headers. In particular,
  the following sender-based algorithms have been added:

  - ``detach``,
  - ``ensure_started``,
  - ``just``,
  - ``just_on``,
  - ``let_error``,
  - ``let_value``,
  - ``on``,
  - ``transform``, and
  - ``when_all``.

  Additionally, futures now implement the sender
  concept. ``make_future`` can be used to turn a sender into a future. All
  functionality is experimental and can change without notice.
- All ``hpx::init`` and ``hpx::start`` overloads now take ``std::function``\ s
  instead of ``hpx::util::function_nonser``. No changes should be required in
  user code to accommodate this change.
- ``hpx::util::unwrapping`` and other related unwrapping functionality has been
  moved up into the ``hpx`` namespace. Names in ``hpx::util`` are still usable
  with a deprecation warning. This functionality can now be accessed through the
  ``hpx/unwrap.hpp`` and ``hpx/local/unwrap.hpp`` headers.
- The default tag for APEX has been update from 2.3.1 to 2.4.0. In particular,
  this fixes a bug which could lead to hangs in distributed runs.
- The dependency on Boost.Asio has been replaced with the standalone Asio
  available at https://github.com/chriskohlhoff/asio. By default, a
  system-installed Asio will be used. ``ASIO_ROOT`` can be given as a hint to
  tell CMake where to find Asio. Alternatively, Asio can be fetched
  automatically  using CMake's fetchcontent by setting
  ``HPX_WITH_FETCH_ASIO=ON``. In general, dependencies on Boost have again been
  reduced.
- Modularization of the library has continued. In this release almost all
  functionality has been moved into modules. These changes do not generally
  affect user code. Warnings are still issued for headers that have moved.
- hipBLAS is now optional when compiling with ``hipcc``. A warning instead of an
  error will be printed if hipBLAS is not found during configuration.
- Previously ``HPX_COMPUTE_HOST_CODE`` was defined in host code only if HPX was
  configured with CUDA or HIP. In this release ``HPX_COMPUTE_HOST_CODE`` is
  always defined in host code.
- An experimental ``HPX_WITH_PRECOMPILED_HEADERS`` CMake option has been added
  to use precompiled headers when building |hpx|. This option should not be used
  on Windows.
- Numerous bug fixes.

Breaking changes
================

- The minimum required CMake version is now 3.17.
- The minimum required Boost version is now 1.71.0.
- The customization mechanism used to implement and extend sender functionality
  and algorithms has been renamed from ``tag_invoke`` to ``tag_dispatch``. All
  customization of sender functionality should be done by overloading
  ``tag_dispatch``.
- The following compatibility options have been removed, along with their
  compatibility implementations:
  - ``HPX_PROGRAM_OPTIONS_WITH_BOOST_PROGRAM_OPTIONS_COMPATIBILITY``
  - ``HPX_WITH_ACTION_BASE_COMPATIBILITY``
  - ``HPX_WITH_EMBEDDED_THREAD_POOLS_COMPATIBILITY``
  - ``HPX_WITH_POOL_EXECUTOR_COMPATIBILITY``.
  - ``HPX_WITH_PROMISE_ALIAS_COMPATIBILITY``
  - ``HPX_WITH_REGISTER_THREAD_COMPATIBILITY``
  - ``HPX_WITH_REGISTER_THREAD_OVERLOADS_COMPATIBILITY``
  - ``HPX_WITH_THREAD_AWARE_TIMER_COMPATIBILITY``
  - ``HPX_WITH_THREAD_EXECUTORS_COMPATIBILITY``
  - ``HPX_WITH_THREAD_POOL_OS_EXECUTOR_COMPATIBILITY``
- The ``HPX_WITH_THREAD_SCHEDULERS`` CMake option has been removed. All
  schedulers are now enabled when possible.
- ``HPX_WITH_INIT_START_OVERLOADS_COMPATIBILITY`` has been turned off by default.

Closed issues
=============

* :hpx-issue:`5423` - Fix lvalue-ref qualified connect for ``when_all-sender``
* :hpx-issue:`5412` - Link error
* :hpx-issue:`5397` - Performance regression in thread annotations
* :hpx-issue:`5395` - HPX 1.7.0-rc1 fails to build icw APEX + OTF2
* :hpx-issue:`5385` - HPX 1.7 crashes on Piz Daint > 64 nodes
* :hpx-issue:`5380` - CMake should search for asio package installed on the
  system
* :hpx-issue:`5378` - HPX 1.7.0 stopped building on Fedora
* :hpx-issue:`5369` - HPX 1.6 and master hangs on Summit for > 64 nodes
* :hpx-issue:`5358` - HPX init fails for single-core environments
* :hpx-issue:`5345` - Rename P2220 property CPOs?
* :hpx-issue:`5333` - HPX does not compile on the new Mac OSX using the M1 chip
* :hpx-issue:`5317` - Consider making hipblas optional
* :hpx-issue:`5306` - asio fails to build with CUDA 10.0
* :hpx-issue:`5294` - ``execution::on`` should be based on
  ``execution::schedule``
* :hpx-issue:`5275` - HPX V1.6.0 fails on Fedora release
* :hpx-issue:`5270` - HPX-1.6.0 fails to build on Windows 10
* :hpx-issue:`5257` - Allow triggering the output of OS thread affinity from
  configuration settings
* :hpx-issue:`5246` - HPX fails to build on ppc64le
* :hpx-issue:`5232` - Annotation using ``hpx::util::annotated_function`` not
  working
* :hpx-issue:`5222` - Build and link errors with ittnotify enabled
* :hpx-issue:`5204` - Move algorithms to tag_fallback_dispatch
* :hpx-issue:`5163` - Remove module-specific compatibility and deprecation
  options
* :hpx-issue:`5161` - Bump required CMake version to 3.17
* :hpx-issue:`5143` - Searching for HPX-Application to generate work on multiple
  Nodes

Closed pull requests
====================

* :hpx-pr:`5438` - Delete datapar/foreach_tests.hpp
* :hpx-pr:`5437` - Add back explicit -pthread flags when available
* :hpx-pr:`5435` - This adds support for systems that assume all types are
  bitwise serializable by default
* :hpx-pr:`5434` - Update CUDA polling logging to be more verbose
* :hpx-pr:`5433` - Fix ``when_all_sender`` connect for references
* :hpx-pr:`5432` - Add deprecation warnings for v1.8
* :hpx-pr:`5431` - Rename the new P0443/P2300 executor to
  ``thread_pool_scheduler``
* :hpx-pr:`5430` - Revert "Adding the missing defined for
  ``HPX_HAVE_DEPRECATION_WARNINGS``"
* :hpx-pr:`5427` - Removing unneeded typedef
* :hpx-pr:`5426` - Adding more concept checks for sender/receiver algorithms
* :hpx-pr:`5425` - Adding the missing defined for
  ``HPX_HAVE_DEPRECATION_WARNINGS``
* :hpx-pr:`5424` - Disable Vc in final docker image created in CI
* :hpx-pr:`5422` - Adding ``execution::experimental::bulk`` algorithm
* :hpx-pr:`5420` - Update logic to find threading library
* :hpx-pr:`5418` - Reduce max size and number of files in ccache cache
* :hpx-pr:`5417` - Final release notes for 1.7.0
* :hpx-pr:`5416` - Adapt ``uninitialized_value_construct`` and
  ``uninitialized_value_construct_n`` to C++ 20
* :hpx-pr:`5415` - Adapt ``uninitialized_default_construct`` and
  ``uninitialized_default_construct_n`` to C++ 20
* :hpx-pr:`5414` - Improve integration of futures and senders
* :hpx-pr:`5413` - Fixing sender/receiver code base to compile with MSVC
* :hpx-pr:`5407` - Handle exceptions thrown during initialization of parcel
  handler
* :hpx-pr:`5406` - Simplify dispatching to annotation handlers
* :hpx-pr:`5405` - Fetch Asio automatically in perftests CI
* :hpx-pr:`5403` - Create generic executor that adds annotations to any other
  executor
* :hpx-pr:`5402` - Adapt ``uninitialized_fill`` and ``uninitialized_fill_n`` to
  C++ 20
* :hpx-pr:`5401` - Modernize a variety of facilities related to parallel
  algorithms
* :hpx-pr:`5400` - Fix sliding semaphore test
* :hpx-pr:`5399` - Rename leftover ``tag_fallback_invoke`` to
  ``tag_fallback_dispatch``
* :hpx-pr:`5398` - Improve logging in AGAS symbol namespace
* :hpx-pr:`5396` - Introduce compatibility layer for collective operations
* :hpx-pr:`5394` - Enable OTF2 in APEX CI configuration
* :hpx-pr:`5393` - Update APEX tag
* :hpx-pr:`5392` - Fixing wrong usage of ``std::forward``
* :hpx-pr:`5391` - Fix forwarding in transform_receiver constructor
* :hpx-pr:`5390` - Make sure shared priority scheduler steals tasks on the
  current NUMA domain when (core) stealing is enabled
* :hpx-pr:`5389` - Adapt ``uninitialized_move`` and ``uninitialized_move_n`` to
  C++ 20
* :hpx-pr:`5388` - Fixing ``gather_there`` for used with lvalue reference
  argument
* :hpx-pr:`5387` - Extend thread state logging and change default stealing
  parameters
* :hpx-pr:`5386` - Attempt to fix the startup hang with nodes > 32
* :hpx-pr:`5384` - Remove HPX 1.5.0 deprecations
* :hpx-pr:`5382` - Prefer installed Asio before considering FetchContent
* :hpx-pr:`5379` - Allow using pre-downloaded (not installed) versions of Asio
  and/or Apex
* :hpx-pr:`5376` - Remove unnecessary explicit listing of library modules.rst
  files in CMakeLists.txt
* :hpx-pr:`5375` - Slight performance improvement for ``hpx::copy`` and
  ``hpx::move`` et.al.
* :hpx-pr:`5374` - Remove unnecessary moves from future sender implementations
* :hpx-pr:`5373` - More changes to clang-cuda Jenkins configuration
* :hpx-pr:`5372` - Slight improvements to ``min/max/minmax_element`` algorithms
* :hpx-pr:`5371` - Adapt ``uninitialized_copy`` and ``uninitialized_copy_n`` to
  C++ 20
* :hpx-pr:`5370` - Decay types in ``just_sender`` ``value_types`` to match
  stored types
* :hpx-pr:`5367` - Disable pkgconfig by default again on macOS
* :hpx-pr:`5365` - Use ccache for Jenkins builds on Piz Daint
* :hpx-pr:`5363` - Update cudatoolkit module name in clang-cuda Jenkins
  configuration
* :hpx-pr:`5362` - Adding ``channel_communicator``
* :hpx-pr:`5361` - Fix compilation with MPI enabled
* :hpx-pr:`5360` - Update APEX and asio tags
* :hpx-pr:`5359` - Fix check for pu-step in single-core case
* :hpx-pr:`5357` - Making sure collective operations can be reused by
  preallocating communicator
* :hpx-pr:`5356` - Update API documentation
* :hpx-pr:`5355` - Make the ``sequenced_executor`` ``processing_units_count``
  member function const
* :hpx-pr:`5354` - Making sure ``default_stack_size`` is defined whenever
  declared
* :hpx-pr:`5353` - Add CUDA timestamp support to HPX Hardware Clock
* :hpx-pr:`5352` - Adding missing includes
* :hpx-pr:`5351` - Adding ``enable_logging/disable_logging`` API functions
* :hpx-pr:`5350` - Adapt lexicographical_compare to C++20
* :hpx-pr:`5349` - Update minimum boost version needed on the docs
* :hpx-pr:`5348` - Rename ``tag_invoke`` and related facilities to
  ``tag_dispatch``
* :hpx-pr:`5347` - Remove ``make_`` prefix for executor properties
* :hpx-pr:`5346` - Remove and disable compatibility options for 1.7.0
* :hpx-pr:`5343` - Fix timed_executor static cast conversion
* :hpx-pr:`5342` - Refactor CUDA event polling
* :hpx-pr:`5341` - Adding ``make_with_annotation`` and ``get_annotation``
  properties
* :hpx-pr:`5339` - Making sure ``hpx::util::hardware::timestamp()`` is always
  defined
* :hpx-pr:`5338` - Fixing ``timed_executor`` specializations of customization
  points
* :hpx-pr:`5335` - Make ``partial_algorithm`` work with any number of arguments
* :hpx-pr:`5334` - Follow up ``iter_sent`` include on #5225
* :hpx-pr:`5332` - Simplify ``tag_invoke`` and friends
* :hpx-pr:`5331` - More work on cleaning up executor CPOs
* :hpx-pr:`5330` - Add option to disable pkgconfig generation
* :hpx-pr:`5328` - Adapt data parallel support using std-simd
* :hpx-pr:`5327` - Fix missing ``ifdef HPX_SMT_PAUSE``
* :hpx-pr:`5326` - Adding ``resize()`` to ``serialize_buffer`` allowing to
  shrink its size
* :hpx-pr:`5324` - Add get member functions to ``async_rw_mutex`` proxy objects
  for explicitly getting the wrapped value
* :hpx-pr:`5323` - Add ``keep_future`` algorithm
* :hpx-pr:`5322` - Replace executor customization point implementations with
  ``tag_invoke``
* :hpx-pr:`5321` - Seperate segmented algorithms for reduce
* :hpx-pr:`5320` - Fix ``is_sender`` trait and other small fixes to p0443 traits
* :hpx-pr:`5319` - gcc 11.1 c++20 build fixes
* :hpx-pr:`5318` - Make hipblas dependency optional as not always available
* :hpx-pr:`5316` - Attempt to fix checking for libatomic
* :hpx-pr:`5315` - Add explicit keyword to fixture constructor
* :hpx-pr:`5314` - Fix a race condition in async mpi affecting limiting executor
* :hpx-pr:`5312` - Use local runtime and local headers in local-only modules and
  tests
* :hpx-pr:`5311` - Add GCC 11 builder to jenkins
* :hpx-pr:`5310` - Adding ``hpx::execution::experimental::task_group``
* :hpx-pr:`5309` - Seperate datapar
* :hpx-pr:`5308` - Seperate segmented algorithms for ``find``, ``find_if``,
  ``find_if_not``
* :hpx-pr:`5307` - Seperate segmented algorithms for ``fill`` and ``generate``
* :hpx-pr:`5304` - Fix compilation of sender CPOs with nvcc
* :hpx-pr:`5300` - Remove ``PRIVATE`` flag that was propagated into the
  ``LANGUAGES``
* :hpx-pr:`5298` - Seperate datapar
* :hpx-pr:`5297` - Specify exact cmake and ninja versions when loading them in
  jenkins jobs
* :hpx-pr:`5295` - Update clang-newest configuration to use clang 12 and Boost
  1.76.0
* :hpx-pr:`5293` - Fix Clang 11 cuda_future test bug
* :hpx-pr:`5292` - Add ``async_rw_mutex`` based on senders
* :hpx-pr:`5291` - "Fix" termination detection
* :hpx-pr:`5290` - Fixed source file line statements in examples documentation
* :hpx-pr:`5289` - Allow splitting of futures holding ``std::tuple``
* :hpx-pr:`5288` - Move algorithms to ``tag_fallback_invoke``
* :hpx-pr:`5287` - Move algorithms to ``tag_fallback_invoke``
* :hpx-pr:`5285` - Fix clang-format failure on master
* :hpx-pr:`5284` - Replacing ``util::function_nonser`` on std::function in
  ``hpx_init``
* :hpx-pr:`5282` - Update Boost for daint 20.11 after update
* :hpx-pr:`5281` - Fix Segmentation fault on ``foreach_datapar_zipiter``
* :hpx-pr:`5280` - Avoid modulo by zero in ``counting_iterator`` test
* :hpx-pr:`5279` - Fix more GCC 10 deprecation warnings
* :hpx-pr:`5277` - Small fixes and improvements to CUDA/MPI polling
* :hpx-pr:`5276` - Fix typo in docs
* :hpx-pr:`5274` - More P1897 algorithms
* :hpx-pr:`5273` - Retry CDash submissions on failure
* :hpx-pr:`5272` - Fix bogus deprecation warnings with GCC 10
* :hpx-pr:`5271` - Correcting target ids for ``symbol_namespace::iterate``
* :hpx-pr:`5268` - Adding generic ``require``, ``require_concept``, and
  ``query`` properties
* :hpx-pr:`5267` - Support annotations in ``hpx::transform_reduce``
* :hpx-pr:`5266` - Making late command line options available for local runtime
* :hpx-pr:`5265` - Leverage ``no_unique_address`` for ``member_pack``
* :hpx-pr:`5264` - Adopt format in more places
* :hpx-pr:`5262` - Install HPX in Rostam Jenkins jobs
* :hpx-pr:`5261` - Limit Rostam Jenkins jobs to marvin partition temporarily
* :hpx-pr:`5260` - Separate segmented algorithms for transform_reduce
* :hpx-pr:`5259` - Making sure late command line options are recognized as
  configuration options
* :hpx-pr:`5258` - Allow for HPX algorithms being invoked with std execution
  policies
* :hpx-pr:`5256` - Separate segmented algorithms for transform
* :hpx-pr:`5255` - Future/sender adapters
* :hpx-pr:`5254` - Fixing datapar
* :hpx-pr:`5253` - Add utility to format ranges
* :hpx-pr:`5252` - Remove uses of Boost.Bimap
* :hpx-pr:`5251` - Banish ``<iostream>`` from library headers
* :hpx-pr:`5250` - Try fixing vc circle ci
* :hpx-pr:`5249` - Adding missing header
* :hpx-pr:`5248` - Use old Piz Daint modules after upgrade
* :hpx-pr:`5247` - Significantly speedup simple ``for_each``, ``for_loop``, and
  ``transform``
* :hpx-pr:`5245` - P1897 ``operator|`` overloads
* :hpx-pr:`5244` - P1897 ``when_all``
* :hpx-pr:`5243` - Make sure ``HPX_DEBUG`` is set based on HPX's build type, not
  consuming project's build type
* :hpx-pr:`5242` - Moving last files unrelated to parcel layer to modules
* :hpx-pr:`5240` - change namespace for ``transform_loop.hpp``
* :hpx-pr:`5238` - Make sure annotations are used in the binary transform
* :hpx-pr:`5237` - Add P1897 ``just``, ``just_on``, and ``on`` algorithms
* :hpx-pr:`5236` - Add an example demonstrating the use of the
  ``invoke_function_action`` facility
* :hpx-pr:`5235` - Attempting to fix datapar compilation issues
* :hpx-pr:`5234` - Fix small typo in ``--hpx:local`` option description
* :hpx-pr:`5233` - Only find Boost.Iostreams if required for plugins
* :hpx-pr:`5231` - Sort printed config options
* :hpx-pr:`5230` - Fix C++20 replace algo adaptation misses
* :hpx-pr:`5229` - Remove leftover Boost include from ``sync_wait.hpp``
* :hpx-pr:`5228` - Print module name only if it has custom configuration
  settings
* :hpx-pr:`5227` - Update .codespell_whitelist
* :hpx-pr:`5226` - Use new docker image in all CircleCI steps
* :hpx-pr:`5225` - Adapt reverse to C++20
* :hpx-pr:`5224` - Separate segmented algorithms for ``none_of``, ``any_of`` and
  ``all_of``
* :hpx-pr:`5223` - Fixing build system for ittnotify
* :hpx-pr:`5221` - Moving LCO related files to modules
* :hpx-pr:`5220` - Seperate segmented algorithms for ``count`` and ``count_if``
* :hpx-pr:`5218` - Seperate segmented algorithms for ``adjacent_find``
* :hpx-pr:`5217` - Add a HIP github action
* :hpx-pr:`5215` - Update ROCm to 4.0.1 on Rostam
* :hpx-pr:`5214` - Fix clang-format error in sender.hpp
* :hpx-pr:`5213` - Removing ESSENTIAL option to the doc example
* :hpx-pr:`5212` - Seperate segmented algorithms for ``for_each_n``
* :hpx-pr:`5211` - Minor adapted algos fixes
* :hpx-pr:`5210` - Fixing ``is_invocable`` deprecation warnings
* :hpx-pr:`5209` - Moving more files into modules (actions, components,
  init_runtime, etc.)
* :hpx-pr:`5208` - Add examples and explanation on when
  ``tag_fallback/priority`` are useful
* :hpx-pr:`5207` - Always define ``HPX_COMPUTE_HOST_CODE`` for host code
* :hpx-pr:`5206` - Add formatting exceptions for libhpx to
  create_module_skeleton.py
* :hpx-pr:`5205` - Moving all distribution policies into modules
* :hpx-pr:`5203` - Move copy algorithms to ``tag_fallback_invoke``
* :hpx-pr:`5202` - Make ``HPX_WITH_PSEUDO_DEPENDENCIES`` a cache variable
* :hpx-pr:`5201` - Replaced ``tag_invoke`` with ``tag_fallback_invoke`` for
  ``adjacent_find`` algorithm
* :hpx-pr:`5200` - Moving files to (distributed) runtime module
* :hpx-pr:`5199` - Update ICC module name on Piz Daint Jenkins configuration
* :hpx-pr:`5198` - Add doxygen documentation for thread_schedule_hint
* :hpx-pr:`5197` - Attempt to fix compilation of context implementations with
  unity build enabled
* :hpx-pr:`5196` - Re-enable component tests
* :hpx-pr:`5195` - Moving files related to colocation logic
* :hpx-pr:`5194` - Another attempt at fixing the Fedora 35 problem
* :hpx-pr:`5193` - Components module
* :hpx-pr:`5192` - Adapt ``replace(_if)`` to C++20
* :hpx-pr:`5190` - Set compatibility headers by default to on
* :hpx-pr:`5188` - Bump Boost minimum version to 1.71.0
* :hpx-pr:`5187` - Force CMake to set the ``-std=c++XX`` flag
* :hpx-pr:`5186` - Remove message to print .cu extension whenever .cu files are
  encountered
* :hpx-pr:`5185` - Remove some minor unnecessary CMake options
* :hpx-pr:`5184` - Remove some leftover ``HPX_WITH_*_SCHEDULER`` uses
* :hpx-pr:`5183` - Remove dependency on boost/iterators/iterator_categories.hpp
* :hpx-pr:`5182` - Fixing Fedora 35 for Power architectures
* :hpx-pr:`5181` - Bump version number and tag post 1.6.0 release
* :hpx-pr:`5180` - Fix htts_v2 tests linking
* :hpx-pr:`5179` - Make sure ``--hpx:local`` command line option is respected
  with networking is off but distributed runtime is on
* :hpx-pr:`5177` - Remove module cmake options
* :hpx-pr:`5176` - Starting to separate segmented algorithms: ``for_each``
* :hpx-pr:`5174` - Don't run segmented algorithms twice on CircleCI
* :hpx-pr:`5173` - Fetching APEX using cmake FetchContent
* :hpx-pr:`5172` - Add separate local-only entry point
* :hpx-pr:`5171` - Remove ``HPX_WITH_THREAD_SCHEDULERS`` CMake option
* :hpx-pr:`5170` - Add ``HPX_WITH_PRECOMPILED_HEADERS`` option
* :hpx-pr:`5166` - Moving some action tests to modules
* :hpx-pr:`5165` - Require cmake 3.17
* :hpx-pr:`5164` - Move ``thread_pool_suspension_helper`` files to small utility
  module
* :hpx-pr:`5160` - Adding checks ensuring modules are not cross-referenced from
  other module categories
* :hpx-pr:`5158` - Replace boost::asio with standalone asio
* :hpx-pr:`5155` - Allow logging when distributed runtime is off
* :hpx-pr:`5153` - Components module
* :hpx-pr:`5152` - Move more files to performance counter module
* :hpx-pr:`5150` - Adapt ``remove_copy(_if)`` to C++20
* :hpx-pr:`5144` - AGAS module
* :hpx-pr:`5125` - Adapt ``remove`` and ``remove_if`` to C++20
* :hpx-pr:`5117` - Attempt to fix segfaults assumed to be caused by
  ``future_data`` instances going out of scope.
* :hpx-pr:`5099` - Allow mixing debug and release builds
* :hpx-pr:`5092` - Replace spirit.qi with x3
* :hpx-pr:`5053` - Add P0443r14 executor and a a few P1897 algorithms
* :hpx-pr:`5044` - Add performance test in jenkins and reports
