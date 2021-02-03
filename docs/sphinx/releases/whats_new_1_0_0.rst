..
    Copyright (C) 2007-2018 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_0_0:

===========================
|hpx| V1.0.0 (Apr 24, 2017)
===========================

General changes
===============

Here are some of the main highlights and changes for this release (in no
particular order):

* Added the facility ``hpx::split_future`` which allows one to convert a
  ``future<tuple<Ts...>>`` into a ``tuple<future<Ts>...>``. This functionality
  is not available when compiling |hpx| with VS2012.
* Added a new type of performance counter which allows one to return a list of
  values for each invocation. We also added a first counter of this type which
  collects a histogram of the times between parcels being created.
* Added new LCOs: ``hpx::lcos::channel`` and ``hpx::lcos::local::channel`` which
  are very similar to the well known channel constructs used in the Go language.
* Added new performance counters reporting the amount of data handled by the
  networking layer on a action-by-action basis (please see :hpx-pr:`2289` for
  more details).
* Added a new facility ``hpx::lcos::barrier``, replacing the equally named older
  one. The new facility has a slightly changed API and is much more efficient.
  Most notable, the new facility exposes a (global) function
  ``hpx::lcos::barrier::synchronize()`` which represents a global barrier across
  all localities.
* We have started to add support for vectorization to our parallel algorithm
  implementations. This support depends on using an external library, currently
  either |vc| or |boost_simd|_. Please see :hpx-issue:`2333` for a list of
  currently supported algorithms. This is an experimental feature and its
  implementation and/or API might change in the future. Please see this
  `blog-post
  <http://stellar-group.org/2016/09/vectorized-cpp-parallel-algorithms-with-hpx/>`_
  for more information.
* The parameter sequence for the ``hpx::parallel::transform_reduce`` overload
  taking one iterator range has changed to match the changes this algorithm has
  undergone while being moved to C++17. The old overload can be still enabled at
  configure time by specifying ``-DHPX_WITH_TRANSFORM_REDUCE_COMPATIBILITY=On``
  to |cmake|.
* The algorithm ``hpx::parallel::inner_product`` has been renamed to
  ``hpx::parallel::transform_reduce`` to match the changes this algorithm has
  undergone while being moved to C++17. The old inner_product names can be still
  enabled at configure time by specifying
  ``-DHPX_WITH_TRANSFORM_REDUCE_COMPATIBILITY=On`` to |cmake|.
* Added versions of ``hpx::get_ptr`` taking client side representations for
  component instances as their parameter (instead of a global id).
* Added the helper utility
  ``hpx::performance_counters::performance_counter_set`` helping to encapsulate
  a set of performance counters to be managed concurrently.
* All execution policies and related classes have been renamed to be consistent
  with the naming changes applied for C++17. All policies now live in the
  namespace ``hpx::parallel::execution``. The ols names can be still enabled at
  configure time by specifying ``-DHPX_WITH_EXECUTION_POLICY_COMPATIBILITY=On``
  to |cmake|.
* The thread scheduling subsystem has undergone a major refactoring which
  results in significant performance improvements. We have also imroved the
  performance of creating ``hpx::future`` and of various facilities handling
  those.
* We have consolidated all of the code in HPX.Compute related to the integration
  of CUDA. ``hpx::partitioned_vector`` has been enabled to be usable with
  ``hpx::compute::vector`` which allows one to place the partitions on one or
  more GPU devices.
* Added new performance counters exposing various internals of the thread
  scheduling subsystem, such as the current idle- and busy-loop counters and
  instantaneous scheduler utilization.
* Extended and improved the use of the ITTNotify hooks allowing to collect
  performance counter data and function annotation information from within the
  Intel Amplifier tool.

Breaking changes
================

* We have dropped support for the gcc compiler versions V4.6 and 4.7. The
  minimal gcc version we now test on is gcc V4.8.
* We have removed (default) support for ``boost::chrono`` in interfaces, uses of
  it have been replaced with ``std::chrono``. This facility can be still enabled
  at configure time by specifying ``-DHPX_WITH_BOOST_CHRONO_COMPATIBILITY=On``
  to |cmake|.
* The parameter sequence for the ``hpx::parallel::transform_reduce`` overload
  taking one iterator range has changed to match the changes this algorithm has
  undergone while being moved to C++17.
* The algorithm ``hpx::parallel::inner_product`` has been renamed to
  ``hpx::parallel::transform_reduce`` to match the changes this algorithm has
  undergone while being moved to C++17.
* the build options ``HPX_WITH_COLOCATED_BACKWARDS_COMPATIBILITY`` and
  ``HPX_WITH_COMPONENT_GET_GID_COMPATIBILITY`` are now disabled by default. Please
  change your code still depending on the deprecated interfaces.

Bug fixes (closed tickets)
==========================

Here is a list of the important tickets we closed for this release.

* :hpx-pr:`2596` - Adding apex data
* :hpx-pr:`2595` - Remove obsolete file
* :hpx-issue:`2594` - FindOpenCL.cmake mismatch with the official cmake module
* :hpx-pr:`2592` - First attempt to introduce spmd_block in hpx
* :hpx-issue:`2591` - Feature request: continuation (then) which does not
  require the callable object to take a future<R> as parameter
* :hpx-pr:`2588` - Daint fixes
* :hpx-pr:`2587` - Fixing transfer_(continuation)_action::schedule
* :hpx-pr:`2585` - Work around MSVC having an ICE when compiling with -Ob2
* :hpx-pr:`2583` - changing 7zip command to 7za in roll_release.sh
* :hpx-pr:`2582` - First attempt to introduce spmd_block in hpx
* :hpx-pr:`2581` - Enable annotated function for parallel algorithms
* :hpx-pr:`2580` - First attempt to introduce spmd_block in hpx
* :hpx-pr:`2579` - Make thread NICE level setting an option
* :hpx-pr:`2578` - Implementing enqueue instead of busy wait when no sender is
  available
* :hpx-pr:`2577` - Retrieve -std=c++11 consistent nvcc flag
* :hpx-pr:`2576` - Add missing dependencies of cuda based tests
* :hpx-pr:`2575` - Remove warnings due to some captured variables
* :hpx-pr:`2573` - Attempt to resolve resolve_locality
* :hpx-pr:`2572` - Adding APEX hooks to background thread
* :hpx-pr:`2571` - Pick up hpx.ignore_batch_env from config map
* :hpx-pr:`2570` - Add commandline options --hpx:print-counters-locally
* :hpx-pr:`2569` - Fix computeapi unit tests
* :hpx-pr:`2567` - This adds another barrier::synchronize before registering
  performance counters
* :hpx-pr:`2564` - Cray static toolchain support
* :hpx-pr:`2563` - Fixed unhandled exception during startup
* :hpx-pr:`2562` - Remove partitioned_vector.cu from build tree when nvcc is
  used
* :hpx-issue:`2561` - octo-tiger crash with commit
  6e921495ff6c26f125d62629cbaad0525f14f7ab
* :hpx-pr:`2560` - Prevent -Wundef warnings on Vc version checks
* :hpx-pr:`2559` - Allowing CUDA callback to set the future directly from an OS
  thread
* :hpx-pr:`2558` - Remove warnings due to float precisions
* :hpx-pr:`2557` - Removing bogus handling of compile flags for CUDA
* :hpx-pr:`2556` - Fixing scan partitioner
* :hpx-pr:`2554` - Add more diagnostics to error thrown from
  find_appropriate_destination
* :hpx-issue:`2555` - No valid parcelport configured
* :hpx-pr:`2553` - Add cmake cuda_arch option
* :hpx-pr:`2552` - Remove incomplete datapar bindings to libflatarray
* :hpx-pr:`2551` - Rename hwloc_topology to hwloc_topology_info
* :hpx-pr:`2550` - Apex api updates
* :hpx-pr:`2549` - Pre-include defines.hpp to get the macro HPX_HAVE_CUDA value
* :hpx-pr:`2548` - Fixing issue with disconnect
* :hpx-pr:`2546` - Some fixes around cuda clang partitioned_vector example
* :hpx-pr:`2545` - Fix uses of the Vc2 datapar flags; the value, not the type,
  should be passed to functions
* :hpx-pr:`2542` - Make HPX_WITH_MALLOC easier to use
* :hpx-pr:`2541` - avoid recompiles when enabling/disabling examples
* :hpx-pr:`2540` - Fixing usage of target_link_libraries()
* :hpx-pr:`2539` - fix RPATH behaviour
* :hpx-issue:`2538` - HPX_WITH_CUDA corrupts compilation flags
* :hpx-pr:`2537` - Add output of a Bazel Skylark extension for paths and compile
  options
* :hpx-pr:`2536` - Add counter exposing total available memory to Windows as
  well
* :hpx-pr:`2535` - Remove obsolete support for security
* :hpx-issue:`2534` - Remove command line option ``--hpx:run-agas-server``
* :hpx-pr:`2533` - Pre-cache locality endpoints during bootstrap
* :hpx-pr:`2532` - Fixing handling of GIDs during serialization preprocessing
* :hpx-pr:`2531` - Amend uses of the term "functor"
* :hpx-pr:`2529` - added counter for reading available memory
* :hpx-pr:`2527` - Facilities to create actions from lambdas
* :hpx-pr:`2526` - Updated docs: HPX_WITH_EXAMPLES
* :hpx-pr:`2525` - Remove warnings related to unused captured variables
* :hpx-issue:`2524` - CMAKE failed because it is missing: TCMALLOC_LIBRARY
  TCMALLOC_INCLUDE_DIR
* :hpx-pr:`2523` - Fixing compose_cb stack overflow
* :hpx-pr:`2522` - Instead of unlocking, ignore the lock while creating the
  message handler
* :hpx-pr:`2521` - Create ``LPROGRESS_`` logging macro to simplify progress
  tracking and timings
* :hpx-pr:`2520` - Intel 17 support
* :hpx-pr:`2519` - Fix components example
* :hpx-pr:`2518` - Fixing parcel scheduling
* :hpx-issue:`2517` - Race condition during Parcel Coalescing Handler creation
* :hpx-issue:`2516` - HPX locks up when using at least 256 localities
* :hpx-issue:`2515` - error: Install cannot find
  "/lib/hpx/libparcel_coalescing.so.0.9.99" but I can see that file
* :hpx-pr:`2514` - Making sure that all continuations of a shared_future are
  invoked in order
* :hpx-pr:`2513` - Fixing locks held during suspension
* :hpx-pr:`2512` - MPI Parcelport improvements and fixes related to the
  background work changes
* :hpx-pr:`2511` - Fixing bit-wise (zero-copy) serialization
* :hpx-issue:`2509` - Linking errors in hwloc_topology
* :hpx-pr:`2508` - Added documentation for debugging with core files
* :hpx-pr:`2506` - Fixing background work invocations
* :hpx-pr:`2505` - Fix tuple serialization
* :hpx-issue:`2504` - Ensure continuations are called in the order they have
  been attached
* :hpx-pr:`2503` - Adding serialization support for Vc v2 (datapar)
* :hpx-pr:`2502` - Resolve various, minor compiler warnings
* :hpx-pr:`2501` - Some other fixes around cuda examples
* :hpx-issue:`2500` - nvcc / cuda clang issue due to a missing -DHPX_WITH_CUDA
  flag
* :hpx-pr:`2499` - Adding support for std::array to wait_all and friends
* :hpx-pr:`2498` - Execute background work as HPX thread
* :hpx-pr:`2497` - Fixing configuration options for spinlock-deadlock detection
* :hpx-pr:`2496` - Accounting for different compilers in CrayKNL toolchain file
* :hpx-pr:`2494` - Adding component base class which ties a component instance
  to a given executor
* :hpx-pr:`2493` - Enable controlling amount of pending threads which must be
  available to allow thread stealing
* :hpx-pr:`2492` - Adding new command line option --hpx:print-counter-reset
* :hpx-pr:`2491` - Resolve ambiguities when compiling with APEX
* :hpx-pr:`2490` - Resuming threads waiting on future with higher priority
* :hpx-issue:`2489` - nvcc issue because -std=c++11 appears twice
* :hpx-pr:`2488` - Adding performance counters exposing the internal idle and
  busy-loop counters
* :hpx-pr:`2487` - Allowing for plain suspend to reschedule thread right away
* :hpx-pr:`2486` - Only flag HPX code for CUDA if HPX_WITH_CUDA is set
* :hpx-pr:`2485` - Making thread-queue parameters runtime-configurable
* :hpx-pr:`2484` - Added atomic counter for parcel-destinations
* :hpx-pr:`2483` - Added priority-queue lifo scheduler
* :hpx-pr:`2482` - Changing scheduler to steal only if more than a minimal
  number of tasks are available
* :hpx-pr:`2481` - Extending command line option --hpx:print-counter-destination
  to support value 'none'
* :hpx-pr:`2479` - Added option to disable signal handler
* :hpx-pr:`2478` - Making sure the sine performance counter module gets loaded
  only for the corresponding example
* :hpx-issue:`2477` - Breaking at a throw statement
* :hpx-pr:`2476` - Annotated function
* :hpx-pr:`2475` - Ensure that using %osthread% during logging will not throw
  for non-hpx threads
* :hpx-pr:`2474` - Remove now superficial non_direct actions from base_lco and
  friends
* :hpx-pr:`2473` - Refining support for ITTNotify
* :hpx-pr:`2472` - Some fixes around hpx compute
* :hpx-issue:`2470` - redefinition of boost::detail::spinlock
* :hpx-issue:`2469` - Dataflow performance issue
* :hpx-pr:`2468` - Perf docs update
* :hpx-pr:`2466` - Guarantee to execute remote direct actions on HPX-thread
* :hpx-pr:`2465` - Improve demo : Async copy and fixed device handling
* :hpx-pr:`2464` - Adding performance counter exposing instantaneous scheduler
  utilization
* :hpx-pr:`2463` - Downcast to future<void>
* :hpx-pr:`2462` - Fixed usage of ITT-Notify API with Intel Amplifier
* :hpx-pr:`2461` - Cublas demo
* :hpx-pr:`2460` - Fixing thread bindings
* :hpx-pr:`2459` - Make -std=c++11 nvcc flag consistent for in-build and
  installed versions
* :hpx-issue:`2457` - Segmentation fault when registering a partitioned vector
* :hpx-pr:`2452` - Properly releasing global barrier for unhandled exceptions
* :hpx-pr:`2451` - Fixing long shutdown times
* :hpx-pr:`2450` - Attempting to fix initialization errors on newer platforms
  (Boost V1.63)
* :hpx-pr:`2449` - Replace BOOST_COMPILER_FENCE with an HPX version
* :hpx-pr:`2448` - This fixes a possible race in the migration code
* :hpx-pr:`2445` - Fixing dataflow et.al. for futures or future-ranges wrapped
                 into ref()
* :hpx-pr:`2444` - Fix segfaults
* :hpx-pr:`2443` - Issue 2442
* :hpx-issue:`2442` - Mismatch between #if/#endif and namespace scope brackets
  in this_thread_executers.hpp
* :hpx-issue:`2441` - undeclared identifier BOOST_COMPILER_FENCE
* :hpx-pr:`2440` - Knl build
* :hpx-pr:`2438` - Datapar backend
* :hpx-pr:`2437` - Adapt algorithm parameter sequence changes from C++17
* :hpx-pr:`2436` - Adapt execution policy name changes from C++17
* :hpx-issue:`2435` - Trunk broken, undefined reference to
  hpx::thread::interrupt(hpx::thread::id, bool)
* :hpx-pr:`2434` - More fixes to resource manager
* :hpx-pr:`2433` - Added versions of ``hpx::get_ptr`` taking client side
  representations
* :hpx-pr:`2432` - Warning fixes
* :hpx-pr:`2431` - Adding facility representing set of performance counters
* :hpx-pr:`2430` - Fix parallel_executor thread spawning
* :hpx-pr:`2429` - Fix attribute warning for gcc
* :hpx-issue:`2427` - Seg fault running octo-tiger with latest HPX commit
* :hpx-issue:`2426` - Bug in 9592f5c0bc29806fce0dbe73f35b6ca7e027edcb causes
  immediate crash in Octo-tiger
* :hpx-pr:`2425` - Fix nvcc errors due to constexpr specifier
* :hpx-issue:`2424` - Async action on component present on hpx::find_here is
  executing synchronously
* :hpx-pr:`2423` - Fix nvcc errors due to constexpr specifier
* :hpx-pr:`2422` - Implementing hpx::this_thread thread data functions
* :hpx-pr:`2421` - Adding benchmark for wait_all
* :hpx-issue:`2420` - Returning object of a component client from another
  component action fails
* :hpx-pr:`2419` - Infiniband parcelport
* :hpx-issue:`2418` - gcc + nvcc fails to compile code that uses
  partitioned_vector
* :hpx-pr:`2417` - Fixing context switching
* :hpx-pr:`2416` - Adding fixes and workarounds to allow compilation with
  nvcc/msvc (VS2015up3)
* :hpx-pr:`2415` - Fix errors coming from hpx compute examples
* :hpx-pr:`2414` - Fixing msvc12
* :hpx-pr:`2413` - Enable cuda/nvcc or cuda/clang when using
  add_hpx_executable()
* :hpx-pr:`2412` - Fix issue in HPX_SetupTarget.cmake when cuda is used
* :hpx-pr:`2411` - This fixes the core compilation issues with MSVC12
* :hpx-issue:`2410` - ``undefined reference to opal_hwloc191_hwloc_.....``
* :hpx-pr:`2409` - Fixing locking for channel and receive_buffer
* :hpx-pr:`2407` - Solving #2402 and #2403
* :hpx-pr:`2406` - Improve guards
* :hpx-pr:`2405` - Enable parallel::for_each for iterators returning proxy types
* :hpx-pr:`2404` - Forward the explicitly given result_type in the hpx invoke
* :hpx-issue:`2403` - datapar_execution + zip iterator: lambda arguments aren't
  references
* :hpx-issue:`2402` - datapar algorithm instantiated with wrong type #2402
* :hpx-pr:`2401` - Added support for imported libraries to HPX_Libraries.cmake
* :hpx-pr:`2400` - Use CMake policy CMP0060
* :hpx-issue:`2399` - Error trying to push back vector of futures to vector
* :hpx-pr:`2398` - Allow config #defines to be written out to custom
  config/defines.hpp
* :hpx-issue:`2397` - CMake generated config defines can cause tedious rebuilds
  category
* :hpx-issue:`2396` - BOOST_ROOT paths are not used at link time
* :hpx-pr:`2395` - Fix target_link_libraries() issue when HPX Cuda is enabled
* :hpx-issue:`2394` - Template compilation error using
  HPX_WITH_DATAPAR_LIBFLATARRAY
* :hpx-pr:`2393` - Fixing lock registration for recursive mutex
* :hpx-pr:`2392` - Add keywords in target_link_libraries in hpx_setup_target
* :hpx-pr:`2391` - Clang goroutines
* :hpx-issue:`2390` - Adapt execution policy name changes from C++17
* :hpx-pr:`2389` - Chunk allocator and pool are not used and are obsolete
* :hpx-pr:`2388` - Adding functionalities to datapar needed by octotiger
* :hpx-pr:`2387` - Fixing race condition for early parcels
* :hpx-issue:`2386` - Lock registration broken for recursive_mutex
* :hpx-pr:`2385` - Datapar zip iterator
* :hpx-pr:`2384` - Fixing race condition in for_loop_reduction
* :hpx-pr:`2383` - Continuations
* :hpx-pr:`2382` - add LibFlatArray-based backend for datapar
* :hpx-pr:`2381` - remove unused typedef to get rid of compiler warnings
* :hpx-pr:`2380` - Tau cleanup
* :hpx-pr:`2379` - Can send immediate
* :hpx-pr:`2378` - Renaming copy_helper/copy_n_helper/move_helper/move_n_helper
* :hpx-issue:`2376` - Boost trunk's spinlock initializer fails to compile
* :hpx-pr:`2375` - Add support for minimal thread local data
* :hpx-pr:`2374` - Adding API functions set_config_entry_callback
* :hpx-pr:`2373` - Add a simple utility for debugging that gives suspended task
  backtraces
* :hpx-pr:`2372` - Barrier Fixes
* :hpx-issue:`2370` - Can't wait on a wrapped future
* :hpx-pr:`2369` - Fixing stable_partition
* :hpx-pr:`2367` - Fixing find_prefixes for Windows platforms
* :hpx-pr:`2366` - Testing for experimental/optional only in C++14 mode
* :hpx-pr:`2364` - Adding set_config_entry
* :hpx-pr:`2363` - Fix papi
* :hpx-pr:`2362` - Adding missing macros for new non-direct actions
* :hpx-pr:`2361` - Improve cmake output to help debug compiler incompatibility
  check
* :hpx-pr:`2360` - Fixing race condition in condition_variable
* :hpx-pr:`2359` - Fixing shutdown when parcels are still in flight
* :hpx-issue:`2357` - failed to insert console_print_action into
  typename_to_id_t registry
* :hpx-pr:`2356` - Fixing return type of get_iterator_tuple
* :hpx-pr:`2355` - Fixing compilation against Boost 1 62
* :hpx-pr:`2354` - Adding serialization for mask_type if CPU_COUNT > 64
* :hpx-pr:`2353` - Adding hooks to tie in APEX into the parcel layer
* :hpx-issue:`2352` - Compile errors when using intel 17 beta (for KNL) on
  edison
* :hpx-pr:`2351` - Fix function vtable get_function_address implementation
* :hpx-issue:`2350` - Build failure - master branch (4de09f5) with Intel
  Compiler v17
* :hpx-pr:`2349` - Enabling zero-copy serialization support for std::vector<>
* :hpx-pr:`2348` - Adding test to verify #2334 is fixed
* :hpx-pr:`2347` - Bug fixes for hpx.compute and hpx::lcos::channel
* :hpx-pr:`2346` - Removing cmake "find" files that are in the APEX cmake
  Modules
* :hpx-pr:`2345` - Implemented parallel::stable_partition
* :hpx-pr:`2344` - Making hpx::lcos::channel usable with basename registration
* :hpx-pr:`2343` - Fix a couple of examples that failed to compile after recent
  api changes
* :hpx-issue:`2342` - Enabling APEX causes link errors
* :hpx-pr:`2341` - Removing cmake "find" files that are in the APEX cmake
  Modules
* :hpx-pr:`2340` - Implemented all existing datapar algorithms using Boost.SIMD
* :hpx-pr:`2339` - Fixing 2338
* :hpx-pr:`2338` - Possible race in sliding semaphore
* :hpx-pr:`2337` - Adjust osu_latency test to measure window_size parcels in
  flight at once
* :hpx-pr:`2336` - Allowing remote direct actions to be executed without
  spawning a task
* :hpx-pr:`2335` - Making sure multiple components are properly initialized from
  arguments
* :hpx-issue:`2334` - Cannot construct component with large vector on a remote
  locality
* :hpx-pr:`2332` - Fixing hpx::lcos::local::barrier
* :hpx-pr:`2331` - Updating APEX support to include OTF2
* :hpx-pr:`2330` - Support for data-parallelism for parallel algorithms
* :hpx-issue:`2329` - Coordinate settings in cmake
* :hpx-pr:`2328` - fix LibGeoDecomp builds with HPX + GCC 5.3.0 + CUDA 8RC
* :hpx-pr:`2326` - Making scan_partitioner work (for now)
* :hpx-issue:`2323` - Constructing a vector of components only correctly
  initializes the first component
* :hpx-pr:`2322` - Fix problems that bubbled up after merging #2278
* :hpx-pr:`2321` - Scalable barrier
* :hpx-pr:`2320` - Std flag fixes
* :hpx-issue:`2319` - -std=c++14 and -std=c++1y with Intel can't build recent
  Boost builds due to insufficient C++14 support; don't enable these flags by
  default for Intel
* :hpx-pr:`2318` - Improve handling of --hpx:bind=<bind-spec>
* :hpx-pr:`2317` - Making sure command line warnings are printed once only
* :hpx-pr:`2316` - Fixing command line handling for default bind mode
* :hpx-pr:`2315` - Set id_retrieved if set_id is present
* :hpx-issue:`2314` - Warning for requested/allocated thread discrepancy is
  printed twice
* :hpx-issue:`2313` - --hpx:print-bind doesn't work with --hpx:pu-step
* :hpx-issue:`2312` - --hpx:bind range specifier restrictions are overly
  restrictive
* :hpx-issue:`2311` - hpx_0.9.99 out of project build fails
* :hpx-pr:`2310` - Simplify function registration
* :hpx-pr:`2309` - Spelling and grammar revisions in documentation (and some
  code)
* :hpx-pr:`2306` - Correct minor typo in the documentation
* :hpx-pr:`2305` - Cleaning up and fixing parcel coalescing
* :hpx-pr:`2304` - Inspect checks for stream related includes
* :hpx-pr:`2303` - Add functionality allowing to enumerate threads of given
  state
* :hpx-pr:`2301` - Algorithm overloads fix for VS2013
* :hpx-pr:`2300` - Use <cstdint>, add inspect checks
* :hpx-pr:`2299` - Replace boost::[c]ref with std::[c]ref, add inspect checks
* :hpx-pr:`2297` - Fixing compilation with no hw_loc
* :hpx-pr:`2296` - Hpx compute
* :hpx-pr:`2295` - Making sure for_loop(execution::par, 0, N, ...) is actually
  executed in parallel
* :hpx-pr:`2294` - Throwing exceptions if the runtime is not up and running
* :hpx-pr:`2293` - Removing unused parcel port code
* :hpx-pr:`2292` - Refactor function vtables
* :hpx-pr:`2291` - Fixing 2286
* :hpx-pr:`2290` - Simplify algorithm overloads
* :hpx-pr:`2289` - Adding performance counters reporting parcel related data on
  a per-action basis
* :hpx-issue:`2288` - Remove dormant parcelports
* :hpx-issue:`2286` - adjustments to parcel handling to support parcelports that
  do not need a connection cache
* :hpx-pr:`2285` - add CMake option to disable package export
* :hpx-pr:`2283` - Add more inspect checks for use of deprecated components
* :hpx-issue:`2282` - Arithmetic exception in executor static chunker
* :hpx-issue:`2281` - For loop doesn't parallelize
* :hpx-pr:`2280` - Fixing 2277: build failure with PAPI
* :hpx-pr:`2279` - Child vs parent stealing
* :hpx-issue:`2277` - master branch build failure (53c5b4f) with papi
* :hpx-pr:`2276` - Compile time launch policies
* :hpx-pr:`2275` - Replace boost::chrono with std::chrono in interfaces
* :hpx-pr:`2274` - Replace most uses of Boost.Assign with initializer list
* :hpx-pr:`2273` - Fixed typos
* :hpx-pr:`2272` - Inspect checks
* :hpx-pr:`2270` - Adding test verifying -Ihpx.os_threads=all
* :hpx-pr:`2269` - Added inspect check for now obsolete boost type traits
* :hpx-pr:`2268` - Moving more code into source files
* :hpx-issue:`2267` - Add inspect support to deprecate Boost.TypeTraits
* :hpx-pr:`2265` - Adding channel LCO
* :hpx-pr:`2264` - Make support for std::ref mandatory
* :hpx-pr:`2263` - Constrain tuple_member forwarding constructor
* :hpx-issue:`2262` - Test hpx.os_threads=all
* :hpx-issue:`2261` - OS X: Error: no matching constructor for initialization of
  'hpx::lcos::local::condition_variable_any'
* :hpx-issue:`2260` - Make support for std::ref mandatory
* :hpx-pr:`2259` - Remove most of Boost.MPL, Boost.EnableIf and Boost.TypeTraits
* :hpx-pr:`2258` - Fixing #2256
* :hpx-pr:`2257` - Fixing launch process
* :hpx-issue:`2256` - Actions are not registered if not invoked
* :hpx-pr:`2255` - Coalescing histogram
* :hpx-pr:`2254` - Silence explicit initialization in copy-constructor warnings
* :hpx-pr:`2253` - Drop support for GCC 4.6 and 4.7
* :hpx-pr:`2252` - Prepare V1.0
* :hpx-pr:`2251` - Convert to 0.9.99
* :hpx-pr:`2249` - Adding iterator_facade and iterator_adaptor
* :hpx-issue:`2248` - Need a feature to yield to a new task immediately
* :hpx-pr:`2246` - Adding split_future
* :hpx-pr:`2245` - Add an example for handing over a component instance to a
  dynamically launched locality
* :hpx-issue:`2243` - Add example demonstrating AGAS symbolic name registration
* :hpx-issue:`2242` - pkgconfig test broken on CentOS 7 / Boost 1.61
* :hpx-issue:`2241` - Compilation error for partitioned vector in hpx_compute
  branch
* :hpx-pr:`2240` - Fixing termination detection on one locality
* :hpx-issue:`2239` - Create a new facility lcos::split_all
* :hpx-issue:`2236` - hpx::cout vs. std::cout
* :hpx-pr:`2232` - Implement local-only primary namespace service
* :hpx-issue:`2147` - would like to know how much data is being routed by
  particular actions
* :hpx-issue:`2109` - Warning while compiling hpx
* :hpx-issue:`1973` - Setting INTERFACE_COMPILE_OPTIONS for hpx_init in CMake
  taints Fortran_FLAGS
* :hpx-issue:`1864` - run_guarded using bound function ignores reference
* :hpx-issue:`1754` - Running with TCP parcelport causes immediate crash or
  freeze
* :hpx-issue:`1655` - Enable zip_iterator to be used with Boost traversal
  iterator categories
* :hpx-issue:`1591` - Optimize AGAS for shared memory only operation
* :hpx-issue:`1401` - Need an efficient infiniband parcelport
* :hpx-issue:`1125` - Fix the IPC parcelport
* :hpx-issue:`839` - Refactor ibverbs and shmem parcelport
* :hpx-issue:`702` - Add instrumentation of parcel layer
* :hpx-issue:`668` - Implement ispc task interface
* :hpx-issue:`533` - Thread queue/deque internal parameters should be runtime
  configurable
* :hpx-issue:`475` - Create a means of combining performance counters into
  querysets

