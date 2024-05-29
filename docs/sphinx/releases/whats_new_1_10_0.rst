..
    Copyright (C) 2007-2024 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_10_0:

============================
|hpx| V1.10.0 (May 29, 2024)
============================

General changes
===============

- The HPX documentation has seen a major overhaul for this release. We finished
  documenting the public local HPX API, we have added migration guides from widely
  used parallelization platforms to HPX (OpenMP, TBB, and MPI).
- We have added facilities enabling optimizations for trivially-relocatable types
  (see |cpp26_p1144|_ for more details).
- We have added (and use) the ``scope_xxx`` helper facilities as specified by the
  C++ library fundamentals TS v3 (see: |cpp26_n4948|_).
- We have added configuration options that allow to build HPX without pre-installing
  any prerequisites. Use ``HPX_WITH_FETCH_HWLOC=On`` to have |hwloc|_ installed for
  you. Similarly, setting ``HPX_WITH_FETCH_BOOST=On`` during configuration time will
  install the necessary |boost|_ libraries (currently V1.84.0).
- We have performed a lot of code cleanup and refactoring to improve the overall
  code quality and decrease compile times.
- The collective operations APIs have seen an unification, we have fixed issues and
  performance problems for the collectives.
- The HPX executors have seen a streamlining and some consistency changes. We have
  applied many performance improvements to the executor implementations that directly
  positively impact the performance of our parallel algorithms.
- We have added a new parcelport allowing to use Gasnet as a communication platform.
- We have added optimizations to various parcelports improving overall communication
  performance. This includes - amongst other things - send immediate optimizations and
  receiver-side zero-copy optimizations.
- Futures will now execute the associated task eagerly and inline on any wait
  operation if the task has not started running yet. This feature can be enabled
  using the ``HPX_COROUTINES_WITH_THREAD_SCHEDULE_HINT_RUNS_AS_CHILD=On`` configuration
  setting (which is `Off` by default).
- We have enabled using json files to supply configuration information through the
  command line. This feature can be enabled with the configuration option
  ``HPX_COMMAND_LINE_HANDLING_WITH_JSON_CONFIGURATION_FILES=On``. This functionality
  depends on the external `JSon library <https://github.com/nlohmann/json>`_, which
  can be built at configuration time by supplying ``HPX_WITH_FETCH_JSON=On`` to
  |cmake|_.
- We have applied many fixes to our CUDA, ROCm, and SYCL build environments.

Breaking changes
================

- The |cmake|_ configuration keys ``SOMELIB_ROOT`` (e.g., ``BOOST_ROOT``) have been
  renamed to ``Somelib_ROOT`` (e.g., ``Boost_ROOT``) to avoid warnings when using
  newer versions of |cmake|_. Please update your scripts accordingly. For now, the
  old variable names are re-assigned to the new names and unset in the |cmake|_
  cache.

Closed issues
=============

* :hpx-issue:`6466` - No access limitations to Wiki
* :hpx-issue:`6461` - handle_received_parcels may never return
* :hpx-issue:`6459` - Building HPX
* :hpx-issue:`6451` - HPX hangs at the very end
* :hpx-issue:`6446` - Issue on page /manual/getting_hpx.html
* :hpx-issue:`6443` - PR #6435 (parcel_layer_tweaks) broke Octo-Tiger
* :hpx-issue:`6440` - HPX does not compile with MSVC of Visual Studio 2022 17.9+
* :hpx-issue:`6437` - HPX 1.9.1 does not compile on Fedora with '#pragma message:  [Parallel STL message]: "Vectorized algorithm unimplemented, redirected to serial
* :hpx-issue:`6419` - Enhancement of the macro functionalities within hpx
* :hpx-issue:`6417` - The current HPX master branch is still not compatible with Kokkos 4.0.1
* :hpx-issue:`6414` - Current HPX master causes segfaults within Octo-Tiger
* :hpx-issue:`6412` - Clangd (Language Server) throws error for __integer_pack at pack.hpp
* :hpx-issue:`6407` - Cannot build Kokkos 4.0.01 with current HPX master
* :hpx-issue:`6405` - Spack Build Error with ROCm 5.7.0
* :hpx-issue:`6398` - HPX sets affinity wrong with multiple processes per node and LCI parcelport enabled
* :hpx-issue:`6392` - [Feature] Install dependencies using CMake
* :hpx-issue:`6388` - HPX error: "Host not found" when running on Expanse with 128 nodes
* :hpx-issue:`6366` - serialize_buffer allocator support needs adjustments
* :hpx-issue:`6361` - HPX 1.9.1 does not compile on Fedora 40
* :hpx-issue:`6355` - Single page documentation is broken
* :hpx-issue:`6334` - Segmentation fault after adding a padding in one_size_heap_list
* :hpx-issue:`6329` - Log hpx threads on forced shutdown
* :hpx-issue:`6316` - Build breaks on FreeBSD
* :hpx-issue:`6299` - HPX does not use distributed localities on Fugaku
* :hpx-issue:`6298` - Update config for coroutines on ARM
* :hpx-issue:`6291` - Zero-copy receive optimization disabled the invocation of direct actions
* :hpx-issue:`6261` - Add optional reading of json files for command line options
* :hpx-issue:`6087` - Support for vcpkg on Linux is broken
* :hpx-issue:`5921` - hpx::info claims that async_mpi was not built, while cmake assures its existence
* :hpx-issue:`5893` - Tests fail on FreeBSD: Executable copyn_test does not exist
* :hpx-issue:`5833` - barrier lockup
* :hpx-issue:`5799` - Investigate CUDA compilation problems
* :hpx-issue:`5340` - Examples do not run on Mac OSX using the M1 chip

Closed pull requests
====================

* :hpx-pr:`6493` - Fix distributed latch documentation
* :hpx-pr:`6492` - Fix kokkos hpx nvcc compilation
* :hpx-pr:`6491` - More fixes to handling bool arguments for collective operations
* :hpx-pr:`6490` - Remove the default max cpu count
* :hpx-pr:`6489` - Ensure TCP parcelport is deactivated if not needed
* :hpx-pr:`6488` - Fixing handling of bool value type for collective operations
* :hpx-pr:`6485` - Destructive interference size
* :hpx-pr:`6484` - Improve performance counter error handling
* :hpx-pr:`6482` - Generalize the notion of bitwise serialization
* :hpx-pr:`6481` - Fixing use of HPX_WITH_CXX_STANDARD
* :hpx-pr:`6480` - Remove equal_to from hpx::any
* :hpx-pr:`6479` - Remove optimizations for certain built-in compiler intrinsics
* :hpx-pr:`6478` - Fixing issues on MacOS
* :hpx-pr:`6477` - lci pp: lci's github repo name changed from LC to lci
* :hpx-pr:`6476` - Fixing binary filter test target names
* :hpx-pr:`6475` - Fix mac os github actions
* :hpx-pr:`6472` - Troubleshoot CI hangs
* :hpx-pr:`6469` - improve(lci pp): more options to control the LCI parcelport
* :hpx-pr:`6467` - Bump jwlawson/actions-setup-cmake from 1.14 to 2.0
* :hpx-pr:`6464` - Update docs of "Writing distributed applications" page
* :hpx-pr:`6463` - Revert "Always return outermost thread id"
* :hpx-pr:`6458` - Reduce test workload to fix CI/CD time-out
* :hpx-pr:`6457` - replace boost::array with std::array and update file name
* :hpx-pr:`6456` - Move APEX CI to rostam
* :hpx-pr:`6455` - Fixing compilation if HPX_HAVE_THREAD_QUEUE_WAITTIME is defined
* :hpx-pr:`6454` - Update perftests reference measurements
* :hpx-pr:`6453` - Update supported platforms of Manual/Prerequisites page
* :hpx-pr:`6452` - Fix nvcc crashes in transform_stream.cu and synchronize.cu
* :hpx-pr:`6450` - Fix git tag name in Getting HPX page
* :hpx-pr:`6449` - LCI parcelport: add yield to potentially infinite retry loop
* :hpx-pr:`6447` - Use compressed ptr in schedulers when 128 atomics are not lockfree
* :hpx-pr:`6445` - Fix agas addressing cache
* :hpx-pr:`6444` - Update CTestConfig.cmake
* :hpx-pr:`6442` - Update CMakeLists.txt
* :hpx-pr:`6441` - Minor documentation fixes
* :hpx-pr:`6439` - Optimizing use of certain #includes
* :hpx-pr:`6438` - Bump jwlawson/actions-setup-cmake from 1.14 to 2.0
* :hpx-pr:`6436` - Update docs
* :hpx-pr:`6435` - Parcel layer tweaks
* :hpx-pr:`6434` - improve termination detection: removing lock from critical path
* :hpx-pr:`6433` - Use shared mutex for resolve_locality procedure
* :hpx-pr:`6432` - Module cleanup up to level 30
* :hpx-pr:`6429` - Making sure HPX_WITH_ASYNC_MPI is reported properly
* :hpx-pr:`6427` - Modifying CMakeLists to copy libhwloc-15.dll to the binary folder in Windows, independently
* :hpx-pr:`6425` - Fix macOS failing test
* :hpx-pr:`6424` - Adding option for downloading Boost using CMake FetchContent
* :hpx-pr:`6423` - Move adjacent_difference to numeric header file
* :hpx-pr:`6422` - Adding steal-half functionalities to work-requesting scheduler
* :hpx-pr:`6421` - Bump actions/checkout from 2 to 4
* :hpx-pr:`6418` - Working around nvcc problems to use CTAD
* :hpx-pr:`6416` - Change run_as_os_thread deprecation forwarding due to hipcc compilation issue
* :hpx-pr:`6415` - Attempting to avoid segfault in OctoTiger during initialization
* :hpx-pr:`6413` - Always return outermost thread id
* :hpx-pr:`6411` - Minor refactoring and fixes to the LCI parcelport and pingpong_performance2 benchmark
* :hpx-pr:`6410` - Adding scope_xxx from library fundamentals TS v3
* :hpx-pr:`6409` - Working around CUDA issue
* :hpx-pr:`6408` - Tightening up collective operation semantics
* :hpx-pr:`6406` - Working around ROCm compiler issue
* :hpx-pr:`6404` - Allow to disable use of [[no_unique_address]] attribute
* :hpx-pr:`6403` - Fixing copyright year
* :hpx-pr:`6402` - fix(lci pp): fix deadlocks with too many failed sends
* :hpx-pr:`6401` - fix(lci pp): fix the null_thread_id bug in the LCI parcelport
* :hpx-pr:`6400` - Fix the affinity setting bug when using LCI pp and multiple localities per node
* :hpx-pr:`6397` - Change API header titles and info
* :hpx-pr:`6396` - Making is_bitwise_serializable SFINAE-friendly
* :hpx-pr:`6395` - Adapt amount of collective testing
* :hpx-pr:`6394` - Adding option for installing Hwloc using CMake FetchContent
* :hpx-pr:`6393` - Optionally disable caching allocator
* :hpx-pr:`6391` - Cleaning up collective operations
* :hpx-pr:`6390` - Making function local constexpr variables non-static
* :hpx-pr:`6389` - Disable resolving hostnames if TCP is disabled
* :hpx-pr:`6387` - Need to break out of the loop when searching the suffixes.
* :hpx-pr:`6384` - Fixing allocation/deallocation mismatch in serialize_buffer
* :hpx-pr:`6383` - Enable fork_join_executor to handle return values from scheduled functions
* :hpx-pr:`6381` - Consistently treat conflicting parameters provided by executors and parameter objects
* :hpx-pr:`6380` - Fixing setting an annotation for an execution policy
* :hpx-pr:`6378` - Allowing to disable signal handlers
* :hpx-pr:`6377` - Fix gasnet-related test failures
* :hpx-pr:`6375` - Update LSU Jenkins with 2023-10 libraries
* :hpx-pr:`6374` - Investigate builder gasnet failure
* :hpx-pr:`6373` - Fixing communicator API, adding docs
* :hpx-pr:`6372` - Fix resource partitioner tests for small thread count
* :hpx-pr:`6371` - Fix jacobi omp examples.
* :hpx-pr:`6370` - improve one_size_heap_list: use rwlock to speedup the allocation/free
* :hpx-pr:`6369` - working issue with MPI_CC / CC conflict in automake
* :hpx-pr:`6368` - Making sure serialize_buffer properly destroys buffer, if needed.
* :hpx-pr:`6367` - Fix parallel relocation test
* :hpx-pr:`6364` - Relocation variants
* :hpx-pr:`6363` - Update the lci parcelport to use LCI v1.7.6
* :hpx-pr:`6362` - Fixing compilation problems on 32 Linux systems
* :hpx-pr:`6360` - Fix broken links in docs: PDF, Single HTML page, Dependency report
* :hpx-pr:`6359` - Fix header file links in Public API page
* :hpx-pr:`6358` - Fix CMake find_library for HWLOC
* :hpx-pr:`6357` - Replace Custom Benchmarking Code with Nanobench
* :hpx-pr:`6356` - Fixed matrix multiplication example output
* :hpx-pr:`6354` - Fix broken links for header files in Public API page
* :hpx-pr:`6353` - Enable using std::reference_wrapper with executor parameters
* :hpx-pr:`6352` - Add Public distributed API documentation
* :hpx-pr:`6350` - Make coverage work with Jenkins Github Branch Source plugin
* :hpx-pr:`6349` - Moving hpx::threads::run_as_xxx to namespace hpx
* :hpx-pr:`6348` - Adding --exclusive to launching tests on rostam
* :hpx-pr:`6346` - changed chat link to discord
* :hpx-pr:`6344` - uninitialized_relocate w/ type_support primitive
* :hpx-pr:`6343` - Bump actions/checkout from 3 to 4
* :hpx-pr:`6342` - Fix HPX-APEX cmake integration
* :hpx-pr:`6341` - Fix shared_future_continuation_order regression test
* :hpx-pr:`6340` - Log alive hpx threads on exit
* :hpx-pr:`6339` - Add coverage testing on Jenkins
* :hpx-pr:`6338` - Fixing HPX_CURRENT_SOURCE_LOCATION when std::source_location exists
* :hpx-pr:`6337` - Remove aurianer, biddisco, and msimberg from codeowners
* :hpx-pr:`6336` - More cleaning up for module levels 19-20
* :hpx-pr:`6335` - Finalize the MPI docs of the Migration Guide
* :hpx-pr:`6332` - More fixes for CMake V3.27
* :hpx-pr:`6330` - Adding basic logging to collective operations
* :hpx-pr:`6328` - Cleanup previous patch adapting to CMake V3.27
* :hpx-pr:`6327` - Modernize modules in level 17 and 18
* :hpx-pr:`6324` - P1144 Relocation primitives
* :hpx-pr:`6321` - Ensure hpx_main is a proper thread_function
* :hpx-pr:`6320` - Fixing cyclic dependencies in naming and agas modules
* :hpx-pr:`6319` - Generate git tag if needed but it is not available
* :hpx-pr:`6317` - Fixing linker problem on FreeBSD
* :hpx-pr:`6315` - acknowledge triv-rel and nothrow-rel types
* :hpx-pr:`6314` - Relocation algorithms Clean
* :hpx-pr:`6313` - Trivial relocation of c-v-ref-array types
* :hpx-pr:`6312` - Fixing warning/error
* :hpx-pr:`6311` - Adding executor parallel invoke CPOs
* :hpx-pr:`6310` - Define HPX_COMPUTE_CODE in builds with SYCL
* :hpx-pr:`6309` - Making sure changed number of cores is propagated to executor
* :hpx-pr:`6308` - openshmem-parcelport initial import
* :hpx-pr:`6306` - The hpxcxx script was broken such that it could only compile for _release
* :hpx-pr:`6305` - Adapting build system for CMake V3.27
* :hpx-pr:`6304` - Fixing an integral type mismatch warning
* :hpx-pr:`6303` - omp for default vectorization
* :hpx-pr:`6301` - Add MPI migration guide
* :hpx-pr:`6294` - Add internal reference counting to semaphores
* :hpx-pr:`6286` - Simd helpers
* :hpx-pr:`6280` - Add TBB to HPX documentation in Migration Guide
* :hpx-pr:`6276` - Add dependabot.yml
* :hpx-pr:`6275` - Revert "Move dependabot.yml into correct directory"
* :hpx-pr:`6272` - set thread name for linux
* :hpx-pr:`6271` - Uninitialised algorithms, move using std::memcpy
* :hpx-pr:`6270` - Bump jwlawson/actions-setup-cmake from 1.9 to 1.14
* :hpx-pr:`6269` - Bump actions/checkout from 2 to 3
* :hpx-pr:`6268` - Move dependabot.yml into correct directory
* :hpx-pr:`6265` - Create dependabot.yml
* :hpx-pr:`6264` - hpx::is_trivially_relocatable trait implementation
* :hpx-pr:`6263` - Adding support for reading json configuration files for command line options
* :hpx-pr:`6249` - Implement the send immediate optimization for the MPI parcelport.
* :hpx-pr:`6237` - Improve compilation performance
* :hpx-pr:`6234` - Adding release notes page for next release
* :hpx-pr:`6233` - Moving is_relocatable to namespace hpx
* :hpx-pr:`6230` - gasnet based parcelport
* :hpx-pr:`6226` - Re-enable dependency on segmented algorithms on CircleCI
* :hpx-pr:`6220` - Add execution on
* :hpx-pr:`6212` - Initial trait definition for `relocatable`
* :hpx-pr:`6199` - added support for unseq, par_unseq for hpx::make_heap algorithm
* :hpx-pr:`6173` - C++ modules
* :hpx-pr:`6122` - Add Module support
* :hpx-pr:`6099` - Futures attempt to execute threads directly if those have not started executing
* :hpx-pr:`6050` - Investigating partitioned_vector problems
* :hpx-pr:`5988` - Adding CI configuration for DGX-A100 at LSU
* :hpx-pr:`5910` - Improve MPI initialization
* :hpx-pr:`5845` - Adding local work requesting scheduler that is based on message passing internally

