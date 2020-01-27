..
    Copyright (C) 2007-2018 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_1_0:

===========================
|hpx| V1.1.0 (Mar 24, 2018)
===========================

General changes
===============

Here are some of the main highlights and changes for this release (in no
particular order):

* We have changed the way |hpx| manages the processing units on a node. We do
  not longer implicitly bind all available cores to a single thread pool. The
  user has now full control over what processing units are bound to what thread
  pool, each with a separate scheduler. It is now also possible to create your
  own scheduler implementation and control what processing units this scheduler
  should use. We added the ``hpx::resource::partitioner`` that manages all
  available processing units and assigns resources to the used thread pools.
  Thread pools can be now be suspended/resumed independently. This functionality
  helps in running |hpx| concurrently to code that is directly relying on
  |openmp|_ and/or |mpi|_.
* We have continued to implement various parallel algorithms. |hpx| now almost
  completely implements all of the parallel algorithms as specified by the
  |cpp17|_. We have also continued to implement these algorithms for the
  distributed use case (for segmented data structures, such as
  ``hpx::partitioned_vector``).
* Added a compatibility layer for ``std::thread``, ``std::mutex``, and
  ``std::condition_variable`` allowing for the code to use those facilities
  where available and to fall back to the corresponding Boost facilities
  otherwise. The |cmake|_ configuration option
  ``-DHPX_WITH_THREAD_COMPATIBILITY=On`` can be used to force using the Boost
  equivalents.
* The parameter sequence for the ``hpx::parallel::transform_inclusive_scan``
  overload taking one iterator range has changed (again) to match the changes
  this algorithm has undergone while being moved to C++17. The old overloads can
  be still enabled at configure time by passing
  ``-DHPX_WITH_TRANSFORM_REDUCE_COMPATIBILITY=On`` to |cmake|_.
* The parameter sequence for the ``hpx::parallel::inclusive_scan`` overload
  taking one iterator range has changed to match the changes this algorithm has
  undergone while being moved to C++17. The old overloads can be still enabled
  at configure time by passing ``-DHPX_WITH_INCLUSIVE_SCAN_COMPATIBILITY=On`` to
  |cmake|.
* Added a helper facility ``hpx::local_new`` which is equivalent to
  ``hpx::new_`` except that it creates components locally only. As a
  consequence, the used component constructor may accept non-serializable
  argument types and/or non-const references or pointers.
* Removed the (broken) component type ``hpx::lcos::queue<T>``. The old type is
  still available at configure time by passing
  ``-DHPX_WITH_QUEUE_COMPATIBILITY=On`` to |cmake|.
* The parallel algorithms adopted for C++17 restrict the iterator categories
  usable with those to at least forward iterators. Our implementation of the
  parallel algorithms was supporting input iterators (and output iterators) as
  well by simply falling back to sequential execution. We have now made our
  implementations conforming by requiring at least forward iterators. In order
  to enable the old behavior use the compatibility option
  ``-DHPX_WITH_ALGORITHM_INPUT_ITERATOR_SUPPORT=On`` on the |cmake|_ command
  line.
* We have added the functionalities allowing for LCOs being implemented using
  (simple) components. Before LCOs had to always be implemented using managed
  components.
* User defined components don't have to be default-constructible anymore. Return
  types from actions don't have to be default-constructible anymore either. Our
  serialization layer now in general supports non-default-constructible types.
* We have added a new launch policy ``hpx::launch::lazy`` that allows oneto
  defer the decision on what launch policy to use to the point of execution.
  This policy is initialized with a function (object) that -- when invoked -- is
  expected to produce the desired launch policy.

Breaking changes
================

* We have dropped support for the gcc compiler version V4.8. The minimal gcc
  version we now test on is gcc V4.9. The minimally required version of |cmake|_
  is now V3.3.2.
* We have dropped support for the Visual Studio 2013 compiler version. The
  minimal Visual Studio version we now test on is Visual Studio 2015.5.
* We have dropped support for the Boost V1.51-V1.54. The minimal version of
  Boost we now test is Boost V1.55.
* We have dropped support for the ``hpx::util::unwrapped`` API.
  ``hpx::util::unwrapped`` will stay functional to some degree, until it finally
  gets removed in a later version of HPX. The functional usage of
  ``hpx::util::unwrapped`` should be changed to the new
  ``hpx::util::unwrapping`` function whereas the immediate usage should be
  replaced to ``hpx::util::unwrap``.
* The performance counter names referring to properties as exposed by the
  threading subsystem have changes as those now additionally have to specify the
  thread-pool. See the corresponding documentation for more details.
* The overloads of ``hpx::async`` that invoke an action do not perform implicit
  unwrapping of the returned future anymore in case the invoked function does
  return a future in the first place. In this case ``hpx::async`` now returns a
  ``hpx::future<future<T>>`` making its behavior conforming to its local
  counterpart.
* We have replaced the use of ``boost::exception_ptr`` in our APIs with the
  equivalent ``std::exception_ptr``. Please change your codes accordingly. No
  compatibility settings are provided.
* We have removed the compatibility settings for
  ``HPX_WITH_COLOCATED_BACKWARDS_COMPATIBILITY`` and
  ``HPX_WITH_COMPONENT_GET_GID_COMPATIBILITY`` as their life-cycle has reached
  its end.
* We have removed the experimental thread schedulers hierarchy_scheduler,
  periodic_priority_scheduler and throttling_scheduler in an effort to clean up
  and consolidate our thread schedulers.

Bug fixes (closed tickets)
==========================

Here is a list of the important tickets we closed for this release.

* :hpx-pr:`3250` - Apex refactoring with guids
* :hpx-pr:`3249` - Updating People.qbk
* :hpx-pr:`3246` - Assorted fixes for CUDA
* :hpx-pr:`3245` - Apex refactoring with guids
* :hpx-pr:`3242` - Modify task counting in thread_queue.hpp
* :hpx-pr:`3240` - Fixed typos
* :hpx-pr:`3238` - Readding accidentally removed std::abort
* :hpx-pr:`3237` - Adding Pipeline example
* :hpx-pr:`3236` - Fixing memory_block
* :hpx-pr:`3233` - Make schedule_thread take suspended threads into account
* :hpx-issue:`3226` - memory_block is breaking, signaling SIGSEGV on a thread on
  creation and freeing
* :hpx-pr:`3225` - Applying quick fix for hwloc-2.0
* :hpx-issue:`3224` - HPX counters crashing the application
* :hpx-pr:`3223` - Fix returns when setting config entries
* :hpx-issue:`3222` - Errors linking libhpx.so
* :hpx-issue:`3221` - HPX on Mac OS X with HWLoc 2.0.0 fails to run
* :hpx-pr:`3216` - Reorder a variadic array to satisfy VS 2017 15.6
* :hpx-pr:`3214` - Changed prerequisites.qbk to avoid confusion while building
  boost
* :hpx-pr:`3213` - Relax locks for thread suspension to avoid holding locks when
  yielding
* :hpx-pr:`3212` - Fix check in sequenced_executor test
* :hpx-pr:`3211` - Use preinit_array to set argc/argv in init_globally example
* :hpx-pr:`3210` - Adapted parallel::{search | search_n} for Ranges TS (see
  #1668)
* :hpx-pr:`3209` - Fix locking problems during shutdown
* :hpx-issue:`3208` - init_globally throwing a run-time error
* :hpx-pr:`3206` - Addition of new arithmetic performance counter "Count"
* :hpx-pr:`3205` - Fixing return type calculation for bulk_then_execute
* :hpx-pr:`3204` - Changing std::rand() to a better inbuilt PRNG generator
* :hpx-pr:`3203` - Resolving problems during shutdown for VS2015
* :hpx-pr:`3202` - Making sure resource partitioner is not accessed if its not
  valid
* :hpx-pr:`3201` - Fixing optional::swap
* :hpx-issue:`3200` - hpx::util::optional fails
* :hpx-pr:`3199` - Fix sliding_semaphore test
* :hpx-pr:`3198` - Set pre_main status before launching run_helper
* :hpx-pr:`3197` - Update README.rst
* :hpx-pr:`3194` - parallel::{fill|fill_n} updated for Ranges TS
* :hpx-pr:`3193` - Updating Runtime.cpp by adding correct description of
  Performance counters during register
* :hpx-pr:`3191` - Fix sliding_semaphore_2338 test
* :hpx-pr:`3190` - Topology improvements
* :hpx-pr:`3189` - Deleting one include of median from BOOST library to
  arithmetics_counter file
* :hpx-pr:`3188` - Optionally disable printing of diagnostics during terminate
* :hpx-pr:`3187` - Suppressing cmake warning issued by cmake > V3.11
* :hpx-pr:`3185` - Remove unused scoped_unlock, unlock_guard_try
* :hpx-pr:`3184` - Fix nqueen example
* :hpx-pr:`3183` - Add runtime start/stop, resume/suspend and OpenMP benchmarks
* :hpx-issue:`3182` - bulk_then_execute has unexpected return type/does not
  compile
* :hpx-issue:`3181` - hwloc 2.0 breaks topo class and cannot be used
* :hpx-issue:`3180` - Schedulers that don't support suspend/resume are unusable
* :hpx-pr:`3179` - Various minor changes to support FLeCSI
* :hpx-pr:`3178` - Fix #3124
* :hpx-pr:`3177` - Removed allgather
* :hpx-pr:`3176` - Fixed Documentation for "using_hpx_pkgconfig"
* :hpx-pr:`3174` - Add hpx::iostreams::ostream overload to format_to
* :hpx-pr:`3172` - Fix lifo queue backend
* :hpx-pr:`3171` - adding the missing unset() function to cpu_mask() for case of
  more than 64 threads
* :hpx-pr:`3170` - Add cmake flag -DHPX_WITH_FAULT_TOLERANCE=ON (OFF by default)
* :hpx-pr:`3169` - Adapted parallel::{count|count_if} for Ranges TS (see #1668)
* :hpx-pr:`3168` - Changing used namespace for seq execution policy
* :hpx-issue:`3167` - Update GSoC projects
* :hpx-issue:`3166` - Application (Octotiger) gets stuck on hpx::finalize when
  only using one thread
* :hpx-issue:`3165` - Compilation of parallel algorithms with HPX_WITH_DATAPAR
  is broken
* :hpx-pr:`3164` - Fixing component migration
* :hpx-pr:`3162` - regex_from_pattern: escape regex special characters to avoid
  misinterpretation
* :hpx-issue:`3161` - Building HPX with hwloc 2.0.0 fails
* :hpx-pr:`3160` - Fixing the handling of quoted command line arguments.
* :hpx-pr:`3158` - Fixing a race with timed suspension (second attempt)
* :hpx-pr:`3157` - Revert "Fixing a race with timed suspension"
* :hpx-pr:`3156` - Fixing serialization of classes with incompatible serialize
  signature
* :hpx-pr:`3154` - More refactorings based on clang-tidy reports
* :hpx-pr:`3153` - Fixing a race with timed suspension
* :hpx-pr:`3152` - Documentation for runtime suspension
* :hpx-pr:`3151` - Use small_vector only from boost version 1.59 onwards
* :hpx-pr:`3150` - Avoiding more stack overflows
* :hpx-pr:`3148` - Refactoring component_base and
  base_action/transfer_base_action
* :hpx-pr:`3147` - Move yield_while out of detail namespace and into own file
* :hpx-pr:`3145` - Remove a leftover of the cxx11 std array cleanup
* :hpx-pr:`3144` - Minor changes to how actions are executed
* :hpx-pr:`3143` - Fix stack overhead
* :hpx-pr:`3142` - Fix typo in config.hpp
* :hpx-pr:`3141` - Fixing small_vector compatibility with older boost version
* :hpx-pr:`3140` - is_heap_text fix
* :hpx-issue:`3139` - Error in is_heap_tests.hpp
* :hpx-pr:`3138` - Partially reverting #3126
* :hpx-pr:`3137` - Suspend speedup
* :hpx-pr:`3136` - Revert "Fixing #2325"
* :hpx-pr:`3135` - Improving destruction of threads
* :hpx-issue:`3134` - HPX_SERIALIZATION_SPLIT_FREE does not stop compiler from
  looking for serialize() method
* :hpx-pr:`3133` - Make hwloc compulsory
* :hpx-pr:`3132` - Update CXX14 constexpr feature test
* :hpx-pr:`3131` - Fixing #2325
* :hpx-pr:`3130` - Avoid completion handler allocation
* :hpx-pr:`3129` - Suspend runtime
* :hpx-pr:`3128` - Make docbook dtd and xsl path names consistent
* :hpx-pr:`3127` - Add hpx::start nullptr overloads
* :hpx-pr:`3126` - Cleaning up coroutine implementation
* :hpx-pr:`3125` - Replacing nullptr with hpx::threads::invalid_thread_id
* :hpx-issue:`3124` - Add hello_world_component to CI builds
* :hpx-pr:`3123` - Add new constructor.
* :hpx-pr:`3122` - Fixing #3121
* :hpx-issue:`3121` - HPX_SMT_PAUSE is broken on non-x86 platforms when __GNUC__
  is defined
* :hpx-pr:`3120` - Don't use boost::intrusive_ptr for thread_id_type
* :hpx-pr:`3119` - Disable default executor compatibility with V1 executors
* :hpx-pr:`3118` - Adding performance_counter::reinit to allow for dynamically
  changing counter sets
* :hpx-pr:`3117` - Replace uses of boost/experimental::optional with
  util::optional
* :hpx-pr:`3116` - Moving background thread APEX timer #2980
* :hpx-pr:`3115` - Fixing race condition in channel test
* :hpx-pr:`3114` - Avoid using util::function for thread function wrappers
* :hpx-pr:`3113` - cmake V3.10.2 has changed the variable names used for MPI
* :hpx-pr:`3112` - Minor fixes to exclusive_scan algorithm
* :hpx-pr:`3111` - Revert "fix detection of cxx11_std_atomic"
* :hpx-pr:`3110` - Suspend thread pool
* :hpx-pr:`3109` - Fixing thread scheduling when yielding a thread id
* :hpx-pr:`3108` - Revert "Suspend thread pool"
* :hpx-pr:`3107` - Remove UB from thread::id relational operators
* :hpx-pr:`3106` - Add cmake test for std::decay_t to fix cuda build
* :hpx-pr:`3105` - Fixing refcount for async traversal frame
* :hpx-pr:`3104` - Local execution of direct actions is now actually performed
  directly
* :hpx-pr:`3103` - Adding support for generic counter_raw_values performance
  counter type
* :hpx-issue:`3102` - Introduce generic performance counter type returning an
  array of values
* :hpx-pr:`3101` - Revert "Adapting stack overhead limit for gcc 4.9"
* :hpx-pr:`3100` - Fix #3068 (condition_variable deadlock)
* :hpx-pr:`3099` - Fixing lock held during suspension in papi counter component
* :hpx-pr:`3098` - Unbreak broadcast_wait_for_2822 test
* :hpx-pr:`3097` - Adapting stack overhead limit for gcc 4.9
* :hpx-pr:`3096` - fix detection of cxx11_std_atomic
* :hpx-pr:`3095` - Add ciso646 header to get _LIBCPP_VERSION for testing inplace
  merge
* :hpx-pr:`3094` - Relax atomic operations on performance counter values
* :hpx-pr:`3093` - Short-circuit all_of/any_of/none_of instantiations
* :hpx-pr:`3092` - Take advantage of C++14 lambda capture initialization syntax,
  where possible
* :hpx-pr:`3091` - Remove more references to Boost from logging code
* :hpx-pr:`3090` - Unify use of yield/yield_k
* :hpx-pr:`3089` - Fix a strange thing in parallel::detail::handle_exception.
  (Fix #2834.)
* :hpx-issue:`3088` - A strange thing in parallel::sort.
* :hpx-pr:`3087` - Fixing assertion in default_distribution_policy
* :hpx-pr:`3086` - Implement parallel::remove and parallel::remove_if
* :hpx-pr:`3085` - Addressing breaking changes in Boost V1.66
* :hpx-pr:`3084` - Ignore build warnings round 2
* :hpx-pr:`3083` - Fix typo HPX_WITH_MM_PREFECTH
* :hpx-pr:`3081` - Pre-decay template arguments early
* :hpx-pr:`3080` - Suspend thread pool
* :hpx-pr:`3079` - Ignore build warnings
* :hpx-pr:`3078` - Don't test inplace_merge with libc++
* :hpx-pr:`3076` - Fixing 3075: Part 1
* :hpx-pr:`3074` - Fix more build warnings
* :hpx-pr:`3073` - Suspend thread cleanup
* :hpx-pr:`3072` - Change existing symbol_namespace::iterate to return all data
  instead of invoking a callback
* :hpx-pr:`3071` - Fixing pack_traversal_async test
* :hpx-pr:`3070` - Fix dynamic_counters_loaded_1508 test by adding dependency to
  memory_component
* :hpx-pr:`3069` - Fix scheduling loop exit
* :hpx-issue:`3068` - hpx::lcos::condition_variable could be suspect to
  deadlocks
* :hpx-pr:`3067` - #ifdef out random_shuffle deprecated in later c++
* :hpx-pr:`3066` - Make coalescing test depend on coalescing library to ensure
  it gets built
* :hpx-pr:`3065` - Workaround for minimal_timed_async_executor_test compilation
  failures, attempts to copy a deferred call (in unevaluated context)
* :hpx-pr:`3064` - Fixing wrong condition in wrapper_heap
* :hpx-pr:`3062` - Fix exception handling for execution::seq
* :hpx-pr:`3061` - Adapt MSVC C++ mode handling to VS15.5
* :hpx-pr:`3060` - Fix compiler problem in MSVC release mode
* :hpx-pr:`3059` - Fixing #2931
* :hpx-issue:`3058` - minimal_timed_async_executor_test_exe fails to compile on
  master (d6f505c)
* :hpx-pr:`3057` - Fix stable_merge_2964 compilation problems
* :hpx-pr:`3056` - Fix some build warnings caused by unused
  variables/unnecessary tests
* :hpx-pr:`3055` - Update documentation for running tests
* :hpx-issue:`3054` - Assertion failure when using bulk hpx::new_ in
  asynchronous mode
* :hpx-pr:`3052` - Do not bind test running to cmake test build rule
* :hpx-pr:`3051` - Fix HPX-Qt interaction in Qt example.
* :hpx-issue:`3048` - nqueen example fails occasionally
* :hpx-pr:`3047` - Fixing #3044
* :hpx-pr:`3046` - Add OS thread suspension
* :hpx-pr:`3042` - PyCicle - first attempt at a build toold for checking PR's
* :hpx-pr:`3041` - Fix a problem about asynchronous execution of parallel::merge
  and parallel::partition.
* :hpx-pr:`3040` - Fix a mistake about exception handling in asynchronous
  execution of scan_partitioner.
* :hpx-pr:`3039` - Consistently use executors to schedule work
* :hpx-pr:`3038` - Fixing local direct function execution and lambda actions
  perfect forwarding
* :hpx-pr:`3035` - Make parallel unit test names match build target/folder names
* :hpx-pr:`3033` - Fix setting of default build type
* :hpx-issue:`3032` - Fix partitioner arg copy found in #2982
* :hpx-issue:`3031` - Errors linking libhpx.so due to missing references (master
  branch, commit 6679a8882)
* :hpx-pr:`3030` - Revert "implement executor then interface with && forwarding
  reference"
* :hpx-pr:`3029` - Run CI inspect checks before building
* :hpx-pr:`3028` - Added range version of parallel::move
* :hpx-issue:`3027` - Implement all scheduling APIs in terms of executors
* :hpx-pr:`3026` - implement executor then interface with && forwarding
  reference
* :hpx-pr:`3025` - Fix typo unitialized to uninitialized
* :hpx-pr:`3024` - Inspect fixes
* :hpx-pr:`3023` - P0356 Simplified partial function application
* :hpx-pr:`3022` - Master fixes
* :hpx-pr:`3021` - Segfault fix
* :hpx-pr:`3020` - Disable command-line aliasing for applications that use
  user_main
* :hpx-pr:`3019` - Adding enable_elasticity option to pool configuration
* :hpx-pr:`3018` - Fix stack overflow detection configuration in header files
* :hpx-pr:`3017` - Speed up local action execution
* :hpx-pr:`3016` - Unify stack-overflow detection options, remove reference to
  libsigsegv
* :hpx-pr:`3015` - Speeding up accessing the resource partitioner and the
  topology info
* :hpx-issue:`3014` - HPX does not compile on POWER8 with gcc 5.4
* :hpx-issue:`3013` - hello_world occasionally prints multiple lines from a
  single OS-thread
* :hpx-pr:`3012` - Silence warning about casting away qualifiers in
  itt_notify.hpp
* :hpx-pr:`3011` - Fix cpuset leak in hwloc_topology_info.cpp
* :hpx-pr:`3010` - Remove useless decay_copy
* :hpx-pr:`3009` - Fixing 2996
* :hpx-pr:`3008` - Remove unused internal function
* :hpx-pr:`3007` - Fixing wrapper_heap alignment problems
* :hpx-issue:`3006` - hwloc memory leak
* :hpx-pr:`3004` - Silence C4251 (needs to have dll-interface) for
  future_data_void
* :hpx-issue:`3003` - Suspension of runtime
* :hpx-pr:`3001` - Attempting to avoid data races in async_traversal while
  evaluating dataflow()
* :hpx-pr:`3000` - Adding hpx::util::optional as a first step to replace
  experimental::optional
* :hpx-pr:`2998` - Cleanup up and Fixing component creation and deletion
* :hpx-issue:`2996` - Build fails with HPX_WITH_HWLOC=OFF
* :hpx-pr:`2995` - Push more future_data functionality to source file
* :hpx-pr:`2994` - WIP: Fix throttle test
* :hpx-pr:`2993` - Making sure --hpx:help does not throw for required (but
  missing) arguments
* :hpx-pr:`2992` - Adding non-blocking (on destruction) service executors
* :hpx-issue:`2991` - run_as_os_thread locks up
* :hpx-issue:`2990` - --help will not work until all required options are
  provided
* :hpx-pr:`2989` - Improve error messages caused by misuse of dataflow
* :hpx-pr:`2988` - Improve error messages caused by misuse of .then
* :hpx-issue:`2987` - stack overflow detection producing false positives
* :hpx-pr:`2986` - Deduplicate non-dependent thread_info logging types
* :hpx-pr:`2985` - Adapted parallel::{all_of|any_of|none_of} for Ranges TS (see
  #1668)
* :hpx-pr:`2984` - Refactor one_size_heap code to simplify code
* :hpx-pr:`2983` - Fixing local_new_component
* :hpx-pr:`2982` - Clang tidy
* :hpx-pr:`2981` - Simplify allocator rebinding in pack traversal
* :hpx-pr:`2979` - Fixing integer overflows
* :hpx-pr:`2978` - Implement parallel::inplace_merge
* :hpx-issue:`2977` - Make hwloc compulsory instead of optional
* :hpx-pr:`2976` - Making sure client_base instance that registered the
  component does not unregister it when being destructed
* :hpx-pr:`2975` - Change version of pulled APEX to master
* :hpx-pr:`2974` - Fix domain not being freed at the end of scheduling loop
* :hpx-pr:`2973` - Fix small typos
* :hpx-pr:`2972` - Adding uintstd.h header
* :hpx-pr:`2971` - Fall back to creating local components using local_new
* :hpx-pr:`2970` - Improve is_tuple_like trait
* :hpx-pr:`2969` - Fix HPX_WITH_MORE_THAN_64_THREADS default value
* :hpx-pr:`2968` - Cleaning up dataflow overload set
* :hpx-pr:`2967` - Make parallel::merge is stable. (Fix #2964.)
* :hpx-pr:`2966` - Fixing a couple of held locks during exception handling
* :hpx-pr:`2965` - Adding missing #include
* :hpx-issue:`2964` - parallel merge is not stable
* :hpx-pr:`2963` - Making sure any function object passed to dataflow is
  released after being invoked
* :hpx-pr:`2962` - Partially reverting #2891
* :hpx-pr:`2961` - Attempt to fix the gcc 4.9 problem with the async pack
  traversal
* :hpx-issue:`2959` - Program terminates during error handling
* :hpx-issue:`2958` - HPX_PLAIN_ACTION breaks due to missing include
* :hpx-pr:`2957` - Fixing errors generated by mixing different attribute
  syntaxes
* :hpx-issue:`2956` - Mixing attribute syntaxes leads to compiler errors
* :hpx-issue:`2955` - Fix OS-Thread throttling
* :hpx-pr:`2953` - Making sure any hpx.os_threads=N supplied through a
  --hpx::config file is taken into account
* :hpx-pr:`2952` - Removing wrong call to cleanup_terminated_locked
* :hpx-pr:`2951` - Revert "Make sure the function vtables are initialized before
  use"
* :hpx-pr:`2950` - Fix a namespace compilation error when some schedulers are
  disabled
* :hpx-issue:`2949` - master branch giving lockups on shutdown
* :hpx-issue:`2947` - hpx.ini is not used correctly at initialization
* :hpx-pr:`2946` - Adding explicit feature test for thread_local
* :hpx-pr:`2945` - Make sure the function vtables are initialized before use
* :hpx-pr:`2944` - Attempting to solve affinity problems on CircleCI
* :hpx-pr:`2943` - Changing channel actions to be direct
* :hpx-pr:`2942` - Adding split_future for std::vector
* :hpx-pr:`2941` - Add a feature test to test for CXX11 override
* :hpx-issue:`2940` - Add split_future for future<vector<T>>
* :hpx-pr:`2939` - Making error reporting during problems with setting affinity
  masks more verbose
* :hpx-pr:`2938` - Fix this various executors
* :hpx-pr:`2937` - Fix some typos in documentation
* :hpx-pr:`2934` - Remove the need for "complete" SFINAE checks
* :hpx-pr:`2933` - Making sure parallel::for_loop is executed in parallel if
  requested
* :hpx-pr:`2932` - Classify chunk_size_iterator to input iterator tag. (Fix
  #2866)
* :hpx-issue:`2931` - --hpx:help triggers unusual error with clang build
* :hpx-pr:`2930` - Add #include files needed to set _POSIX_VERSION for debug
  check
* :hpx-pr:`2929` - Fix a couple of deprecated c++ features
* :hpx-pr:`2928` - Fixing execution parameters
* :hpx-issue:`2927` - CMake warning: ... cycle in constraint graph
* :hpx-pr:`2926` - Default pool rename
* :hpx-issue:`2925` - Default pool cannot be renamed
* :hpx-issue:`2924` - hpx:attach-debugger=startup does not work any more
* :hpx-pr:`2923` - Alloc membind
* :hpx-pr:`2922` - This fixes CircleCI errors when running with --hpx:bind=none
* :hpx-pr:`2921` - Custom pool executor was missing priority and stacksize
  options
* :hpx-pr:`2920` - Adding test to trigger problem reported in #2916
* :hpx-pr:`2919` - Make sure the resource_partitioner is properly destructed on
  hpx::finalize
* :hpx-issue:`2918` - hpx::init calls wrong (first) callback when called
  multiple times
* :hpx-pr:`2917` - Adding util::checkpoint
* :hpx-issue:`2916` - Weird runtime failures when using a channel and chained
  continuations
* :hpx-pr:`2915` - Introduce executor parameters customization points
* :hpx-issue:`2914` - Task assignment to current Pool has unintended
  consequences
* :hpx-pr:`2913` - Fix rp hang
* :hpx-pr:`2912` - Update contributors
* :hpx-pr:`2911` - Fixing CUDA problems
* :hpx-pr:`2910` - Improve error reporting for process component on POSIX
  systems
* :hpx-pr:`2909` - Fix typo in include path
* :hpx-pr:`2908` - Use proper container according to iterator tag in benchmarks
  of parallel algorithms
* :hpx-pr:`2907` - Optionally force-delete remaining channel items on close
* :hpx-pr:`2906` - Making sure generated performance counter names are correct
* :hpx-issue:`2905` - collecting idle-rate performance counters on multiple
  localities produces an error
* :hpx-issue:`2904` - build broken for Intel 17 compilers
* :hpx-pr:`2903` - Documentation Updates-- Adding New People
* :hpx-pr:`2902` - Fixing service_executor
* :hpx-pr:`2901` - Fixing partitioned_vector creation
* :hpx-pr:`2900` - Add numa-balanced mode to hpx::bind, spread cores over numa
  domains
* :hpx-issue:`2899` - hpx::bind does not have a mode that balances cores over
  numa domains
* :hpx-pr:`2898` - Adding missing #include and missing guard for optional code
  section
* :hpx-pr:`2897` - Removing dependency on Boost.ICL
* :hpx-issue:`2896` - Debug build fails without -fpermissive with GCC 7.1 and
  Boost 1.65
* :hpx-pr:`2895` - Fixing SLURM environment parsing
* :hpx-pr:`2894` - Fix incorrect handling of compile definition with value 0
* :hpx-issue:`2893` - Disabling schedulers causes build errors
* :hpx-pr:`2892` - added list serializer
* :hpx-pr:`2891` - Resource Partitioner Fixes
* :hpx-issue:`2890` - Destroying a non-empty channel causes an assertion failure
* :hpx-pr:`2889` - Add check for libatomic
* :hpx-pr:`2888` - Fix compilation problems if HPX_WITH_ITT_NOTIFY=ON
* :hpx-pr:`2887` - Adapt broadcast() to non-unwrapping async<Action>
* :hpx-pr:`2886` - Replace Boost.Random with C++11 <random>
* :hpx-issue:`2885` - regression in broadcast?
* :hpx-issue:`2884` - linking ``-latomic`` is not portable
* :hpx-pr:`2883` - Explicitly set -pthread flag if available
* :hpx-pr:`2882` - Wrap boost::format uses
* :hpx-issue:`2881` - hpx not compiling with ``HPX_WITH_ITTNOTIFY=On``
* :hpx-issue:`2880` - hpx::bind scatter/balanced give wrong pu masks
* :hpx-pr:`2878` - Fix incorrect pool usage masks setup in RP/thread manager
* :hpx-pr:`2877` - Require ``std::array`` by default
* :hpx-pr:`2875` - Deprecate use of BOOST_ASSERT
* :hpx-pr:`2874` - Changed serialization of boost.variant to use variadic
  templates
* :hpx-issue:`2873` - building with parcelport_mpi fails on cori
* :hpx-pr:`2871` - Adding missing support for throttling scheduler
* :hpx-pr:`2870` - Disambiguate use of base_lco_with_value macros with channel
* :hpx-issue:`2869` - Difficulty compiling
  ``HPX_REGISTER_CHANNEL_DECLARATION(double)``
* :hpx-pr:`2868` - Removing unneeded assert
* :hpx-pr:`2867` - Implement parallel::unique
* :hpx-issue:`2866` - The chunk_size_iterator violates multipass guarantee
* :hpx-pr:`2865` - Only use sched_getcpu on linux machines
* :hpx-pr:`2864` - Create redistribution archive for successful builds
* :hpx-pr:`2863` - Replace casts/assignments with hard-coded memcpy operations
* :hpx-issue:`2862` - sched_getcpu not available on MacOS
* :hpx-pr:`2861` - Fixing unmatched header defines and recursive inclusion of
  threadmanager
* :hpx-issue:`2860` - Master program fails with assertion 'type ==
  data_type_address' failed: HPX(assertion_failure)
* :hpx-issue:`2852` - Support for ARM64
* :hpx-pr:`2858` - Fix misplaced #if #endif's that cause build failure without
  THREAD_CUMULATIVE_COUNTS
* :hpx-pr:`2857` - Fix some listing in documentation
* :hpx-pr:`2856` - Fixing component handling for lcos
* :hpx-pr:`2855` - Add documentation for coarrays
* :hpx-pr:`2854` - Support ARM64 in timestamps
* :hpx-pr:`2853` - Update Table 17. Non-modifying Parallel Algorithms in
  Documentation
* :hpx-pr:`2851` - Allowing for non-default-constructible component types
* :hpx-pr:`2850` - Enable returning future<R> from actions where R is not
  default-constructible
* :hpx-pr:`2849` - Unify serialization of non-default-constructable types
* :hpx-issue:`2848` - Components have to be default constructible
* :hpx-issue:`2847` - Returning a future<R> where R is not default-constructable
  broken
* :hpx-issue:`2846` - Unify serialization of non-default-constructible types
* :hpx-pr:`2845` - Add Visual Studio 2015 to the tested toolchains in Appveyor
* :hpx-issue:`2844` - Change the appveyor build to use the minimal required MSVC
  version
* :hpx-issue:`2843` - multi node hello_world hangs
* :hpx-pr:`2842` - Correcting Spelling mistake in docs
* :hpx-pr:`2841` - Fix usage of std::aligned_storage
* :hpx-pr:`2840` - Remove constexpr from a void function
* :hpx-issue:`2839` - memcpy buffer overflow: load_construct_data() and
  std::complex members
* :hpx-issue:`2835` - ``constexpr`` functions with ``void`` return type break
  compilation with CUDA 8.0
* :hpx-issue:`2834` - One suspicion in parallel::detail::handle_exception
* :hpx-pr:`2833` - Implement parallel::merge
* :hpx-pr:`2832` - Fix a strange thing in
  parallel::util::detail::handle_local_exceptions. (Fix #2818)
* :hpx-pr:`2830` - Break the debugger when a test failed
* :hpx-issue:`2831` - ``parallel/executors/execution_fwd.hpp`` causes
  compilation failure in C++11 mode.
* :hpx-pr:`2829` - Implement an API for asynchronous pack traversal
* :hpx-pr:`2828` - Split unit test builds on CircleCI to avoid timeouts
* :hpx-issue:`2827` - failure to compile hello_world example with -Werror
* :hpx-pr:`2824` - Making sure promises are marked as started when used as
  continuations
* :hpx-pr:`2823` - Add documentation for partitioned_vector_view
* :hpx-issue:`2822` - Yet another issue with wait_for similar to #2796
* :hpx-pr:`2821` - Fix bugs and improve that about
  HPX_HAVE_CXX11_AUTO_RETURN_VALUE of CMake
* :hpx-pr:`2820` - Support C++11 in benchmark codes of parallel::partition and
  parallel::partition_copy
* :hpx-pr:`2819` - Fix compile errors in unit test of container version of
  parallel::partition
* :hpx-issue:`2818` - A strange thing in
  parallel::util::detail::handle_local_exceptions
* :hpx-issue:`2815` - HPX fails to compile with HPX_WITH_CUDA=ON and the new
  CUDA 9.0 RC
* :hpx-issue:`2814` - Using 'gmakeN' after 'cmake' produces error in
  src/CMakeFiles/hpx.dir/runtime/agas/addressing_service.cpp.o
* :hpx-pr:`2813` - Properly support [[noreturn]] attribute if available
* :hpx-issue:`2812` - Compilation fails with gcc 7.1.1
* :hpx-pr:`2811` - Adding hpx::launch::lazy and support for async, dataflow, and
  future::then
* :hpx-pr:`2810` - Add option allowing to disable deprecation warning
* :hpx-pr:`2809` - Disable throttling scheduler if HWLOC is not found/used
* :hpx-pr:`2808` - Fix compile errors on some environments of
  parallel::partition
* :hpx-issue:`2807` - Difficulty building with ``HPX_WITH_HWLOC=Off``
* :hpx-pr:`2806` - Partitioned vector
* :hpx-pr:`2805` - Serializing collections with non-default constructible data
* :hpx-pr:`2802` - Fix FreeBSD 11
* :hpx-issue:`2801` - Rate limiting techniques in io_service
* :hpx-issue:`2800` - New Launch Policy: async_if
* :hpx-pr:`2799` - Fix a unit test failure on GCC in tuple_cat
* :hpx-pr:`2798` - bump minimum required cmake to 3.0 in test
* :hpx-pr:`2797` - Making sure future::wait_for et.al. work properly for action
  results
* :hpx-issue:`2796` - wait_for does always in "deferred" state for calls on
  remote localities
* :hpx-issue:`2795` - Serialization of types without default constructor
* :hpx-pr:`2794` - Fixing test for partitioned_vector iteration
* :hpx-pr:`2792` - Implemented segmented find and its variations for partitioned
  vector
* :hpx-pr:`2791` - Circumvent scary warning about placement new
* :hpx-pr:`2790` - Fix OSX build
* :hpx-pr:`2789` - Resource partitioner
* :hpx-pr:`2788` - Adapt parallel::is_heap and parallel::is_heap_until to Ranges
  TS
* :hpx-pr:`2787` - Unwrap hotfixes
* :hpx-pr:`2786` - Update CMake Minimum Version to 3.3.2 (refs #2565)
* :hpx-issue:`2785` - Issues with masks and cpuset
* :hpx-pr:`2784` - Error with reduce and transform reduce fixed
* :hpx-pr:`2783` - StackOverflow integration with libsigsegv
* :hpx-pr:`2782` - Replace boost::atomic with std::atomic (where possible)
* :hpx-pr:`2781` - Check for and optionally use [[deprecated]] attribute
* :hpx-pr:`2780` - Adding empty (but non-trivial) destructor to circumvent
  warnings
* :hpx-pr:`2779` - Exception info tweaks
* :hpx-pr:`2778` - Implement parallel::partition
* :hpx-pr:`2777` - Improve error handling in gather_here/gather_there
* :hpx-pr:`2776` - Fix a bug in compiler version check
* :hpx-pr:`2775` - Fix compilation when HPX_WITH_LOGGING is OFF
* :hpx-pr:`2774` - Removing dependency on Boost.Date_Time
* :hpx-pr:`2773` - Add sync_images() method to spmd_block class
* :hpx-pr:`2772` - Adding documentation for PAPI counters
* :hpx-pr:`2771` - Removing boost preprocessor dependency
* :hpx-pr:`2770` - Adding test, fixing deadlock in config registry
* :hpx-pr:`2769` - Remove some other warnings and errors detected by clang 5.0
* :hpx-issue:`2768` - Is there iterator tag for HPX?
* :hpx-pr:`2767` - Improvements to continuation annotation
* :hpx-pr:`2765` - gcc split stack support for HPX threads #620
* :hpx-pr:`2764` - Fix some uses of begin/end, remove unnecessary includes
* :hpx-pr:`2763` - Bump minimal Boost version to 1.55.0
* :hpx-pr:`2762` - hpx::partitioned_vector serializer
* :hpx-pr:`2761` - Adding configuration summary to cmake output and --hpx:info
* :hpx-pr:`2760` - Removing 1d_hydro example as it is broken
* :hpx-pr:`2758` - Remove various warnings detected by clang 5.0
* :hpx-issue:`2757` - In case of a "raw thread" is needed per core for
  implementing parallel algorithm, what is good practice in HPX?
* :hpx-pr:`2756` - Allowing for LCOs to be simple components
* :hpx-pr:`2755` - Removing make_index_pack_unrolled
* :hpx-pr:`2754` - Implement parallel::unique_copy
* :hpx-pr:`2753` - Fixing detection of [[fallthrough]] attribute
* :hpx-pr:`2752` - New thread priority names
* :hpx-pr:`2751` - Replace boost::exception with proposed exception_info
* :hpx-pr:`2750` - Replace boost::iterator_range
* :hpx-pr:`2749` - Fixing hdf5 examples
* :hpx-issue:`2748` - HPX fails to build with enabled hdf5 examples
* :hpx-issue:`2747` - Inherited task priorities break certain DAG optimizations
* :hpx-issue:`2746` - HPX segfaulting with valgrind
* :hpx-pr:`2745` - Adding extended arithmetic performance counters
* :hpx-pr:`2744` - Adding ability to statistics counters to reset base counter
* :hpx-issue:`2743` - Statistics counter does not support resetting
* :hpx-pr:`2742` - Making sure Vc V2 builds without additional HPX configuration
  flags
* :hpx-pr:`2741` - Deprecate unwrapped and implement unwrap and unwrapping
* :hpx-pr:`2740` - Coroutine stackoverflow detection for linux/posix; Issue
  #2408
* :hpx-pr:`2739` - Add files via upload
* :hpx-pr:`2738` - Appveyor support
* :hpx-pr:`2737` - Fixing 2735
* :hpx-issue:`2736` - 1d_hydro example doesn't work
* :hpx-issue:`2735` - partitioned_vector_subview test failing
* :hpx-pr:`2734` - Add C++11 range utilities
* :hpx-pr:`2733` - Adapting iterator requirements for parallel algorithms
* :hpx-pr:`2732` - Integrate C++ Co-arrays
* :hpx-pr:`2731` - Adding on_migrated event handler to migratable component
  instances
* :hpx-issue:`2729` - Add on_migrated() event handler to migratable components
* :hpx-issue:`2728` - Why Projection is needed in parallel algorithms?
* :hpx-pr:`2727` - Cmake files for StackOverflow Detection
* :hpx-pr:`2726` - CMake for Stack Overflow Detection
* :hpx-pr:`2725` - Implemented segmented algorithms for partitioned vector
* :hpx-pr:`2724` - Fix examples in Action documentation
* :hpx-pr:`2723` - Enable lcos::channel<T>::register_as
* :hpx-issue:`2722` - channel register_as() failing on compilation
* :hpx-pr:`2721` - Mind map
* :hpx-pr:`2720` - reorder forward declarations to get rid of C++14-only auto
  return types
* :hpx-pr:`2719` - Add documentation for partitioned_vector and add features in
  pack.hpp
* :hpx-issue:`2718` - Some forward declarations in execution_fwd.hpp aren't
  C++11-compatible
* :hpx-pr:`2717` - Config support for fallthrough attribute
* :hpx-pr:`2716` - Implement parallel::partition_copy
* :hpx-pr:`2715` - initial import of icu string serializer
* :hpx-pr:`2714` - initial import of valarray serializer
* :hpx-pr:`2713` - Remove slashes before CMAKE_FILES_DIRECTORY variables
* :hpx-pr:`2712` - Fixing wait for 1751
* :hpx-pr:`2711` - Adjust code for minimal supported GCC having being bumped to
  4.9
* :hpx-pr:`2710` - Adding code of conduct
* :hpx-pr:`2709` - Fixing UB in destroy tests
* :hpx-pr:`2708` - Add inline to prevent multiple definition issue
* :hpx-issue:`2707` - Multiple defined symbols for task_block.hpp in VS2015
* :hpx-pr:`2706` - Adding .clang-format file
* :hpx-pr:`2704` - Add a synchronous mapping API
* :hpx-issue:`2703` - Request: Add the .clang-format file to the repository
* :hpx-issue:`2702` - STEllAR-GROUP/Vc slower than VCv1 possibly due to wrong
  instructions generated
* :hpx-issue:`2701` - Datapar with STEllAR-GROUP/Vc requires obscure flag
* :hpx-issue:`2700` - Naming inconsistency in parallel algorithms
* :hpx-issue:`2699` - Iterator requirements are different from standard in
  parallel copy_if.
* :hpx-pr:`2698` - Properly releasing parcelport write handlers
* :hpx-issue:`2697` - Compile error in addressing_service.cpp
* :hpx-issue:`2696` - Building and using HPX statically: undefined references
  from runtime_support_server.cpp
* :hpx-issue:`2695` - Executor changes cause compilation failures
* :hpx-pr:`2694` - Refining C++ language mode detection for MSVC
* :hpx-pr:`2693` - P0443 r2
* :hpx-pr:`2692` - Partially reverting changes to parcel_await
* :hpx-issue:`2689` - HPX build fails when HPX_WITH_CUDA is enabled
* :hpx-pr:`2688` - Make Cuda Clang builds pass
* :hpx-pr:`2687` - Add an is_tuple_like trait for sequenceable type detection
* :hpx-pr:`2686` - Allowing throttling scheduler to be used without idle backoff
* :hpx-pr:`2685` - Add support of std::array to hpx::util::tuple_size and
  tuple_element
* :hpx-pr:`2684` - Adding new statistics performance counters
* :hpx-pr:`2683` - Replace boost::exception_ptr with std::exception_ptr
* :hpx-issue:`2682` - HPX does not compile with
  HPX_WITH_THREAD_MANAGER_IDLE_BACKOFF=OFF
* :hpx-pr:`2681` - Attempt to fix problem in managed_component_base
* :hpx-pr:`2680` - Fix bad size during archive creation
* :hpx-issue:`2679` - Mismatch between size of archive and container
* :hpx-issue:`2678` - In parallel algorithm, other tasks are executed to the end
  even if an exception occurs in any task.
* :hpx-pr:`2677` - Adding include check for std::addressof
* :hpx-pr:`2676` - Adding parallel::destroy and destroy_n
* :hpx-pr:`2675` - Making sure statistics counters work as expected
* :hpx-pr:`2674` - Turning assertions into exceptions
* :hpx-pr:`2673` - Inhibit direct conversion from future<future<T>> -->
  future<void>
* :hpx-pr:`2672` - C++17 invoke forms
* :hpx-pr:`2671` - Adding uninitialized_value_construct and
  uninitialized_value_construct_n
* :hpx-pr:`2670` - Integrate spmd multidimensional views for
  partitioned_vectors
* :hpx-pr:`2669` - Adding uninitialized_default_construct and
  uninitialized_default_construct_n
* :hpx-pr:`2668` - Fixing documentation index
* :hpx-issue:`2667` - Ambiguity of nested hpx::future<void>'s
* :hpx-issue:`2666` - Statistics Performance counter is not working
* :hpx-pr:`2664` - Adding uninitialized_move and uninitialized_move_n
* :hpx-issue:`2663` - Seg fault in managed_component::get_base_gid, possibly
  cause by util::reinitializable_static
* :hpx-issue:`2662` - Crash in managed_component::get_base_gid due to problem
  with util::reinitializable_static
* :hpx-pr:`2665` - Hide the ``detail`` namespace in doxygen per default
* :hpx-pr:`2660` - Add documentation to hpx::util::unwrapped and
  hpx::util::unwrapped2
* :hpx-pr:`2659` - Improve integration with vcpkg
* :hpx-pr:`2658` - Unify access_data trait for use in both, serialization and
  de-serialization
* :hpx-pr:`2657` - Removing hpx::lcos::queue<T>
* :hpx-pr:`2656` - Reduce MAX_TERMINATED_THREADS default, improve memory use on
  manycore cpus
* :hpx-pr:`2655` - Mainteinance for emulate-deleted macros
* :hpx-pr:`2654` - Implement parallel is_heap and is_heap_until
* :hpx-pr:`2653` - Drop support for VS2013
* :hpx-pr:`2652` - This patch makes sure that all parcels in a batch are
  properly handled
* :hpx-pr:`2649` - Update docs (Table 18) - move transform to end
* :hpx-issue:`2647` - hpx::parcelset::detail::parcel_data::has_continuation_ is
  uninitialized
* :hpx-issue:`2644` - Some .vcxproj in the HPX.sln fail to build
* :hpx-issue:`2641` - ``hpx::lcos::queue`` should be deprecated
* :hpx-pr:`2640` - A new throttling policy with public APIs to suspend/resume
* :hpx-pr:`2639` - Fix a tiny typo in tutorial.
* :hpx-issue:`2638` - Invalid return type 'void' of constexpr function
* :hpx-pr:`2636` - Add and use HPX_MSVC_WARNING_PRAGMA for #pragma warning
* :hpx-pr:`2633` - Distributed define_spmd_block
* :hpx-pr:`2632` - Making sure container serialization uses size-compatible
  types
* :hpx-pr:`2631` - Add lcos::local::one_element_channel
* :hpx-pr:`2629` - Move unordered_map out of parcelport into hpx/concurrent
* :hpx-pr:`2628` - Making sure that shutdown does not hang
* :hpx-pr:`2627` - Fix serialization
* :hpx-pr:`2626` - Generate ``cmake_variables.qbk`` and ``cmake_toolchains.qbk``
  outside of the source tree
* :hpx-pr:`2625` - Supporting -std=c++17 flag
* :hpx-pr:`2624` - Fixing a small cmake typo
* :hpx-pr:`2622` - Update CMake minimum required version to 3.0.2 (closes #2621)
* :hpx-issue:`2621` - Compiling hpx master fails with /usr/bin/ld: final link
  failed: Bad value
* :hpx-pr:`2620` - Remove warnings due to some captured variables
* :hpx-pr:`2619` - LF multiple parcels
* :hpx-pr:`2618` - Some fixes to libfabric that didn't get caught before the
  merge
* :hpx-pr:`2617` - Adding ``hpx::local_new``
* :hpx-pr:`2616` - Documentation: Extract all entities in order to autolink
  functions correctly
* :hpx-issue:`2615` - Documentation: Linking functions is broken
* :hpx-pr:`2614` - Adding serialization for std::deque
* :hpx-pr:`2613` - We need to link with boost.thread and boost.chrono if we use
  boost.context
* :hpx-pr:`2612` - Making sure for_loop_n(par, ...) is actually executed in
  parallel
* :hpx-pr:`2611` - Add documentation to invoke_fused and friends NFC
* :hpx-pr:`2610` - Added reduction templates using an identity value
* :hpx-pr:`2608` - Fixing some unused vars in inspect
* :hpx-pr:`2607` - Fixed build for mingw
* :hpx-pr:`2606` - Supporting generic context for boost >= 1.61
* :hpx-pr:`2605` - Parcelport libfabric3
* :hpx-pr:`2604` - Adding allocator support to promise and friends
* :hpx-pr:`2603` - Barrier hang
* :hpx-pr:`2602` - Changes to scheduler to steal from one high-priority queue
* :hpx-issue:`2601` - High priority tasks are not executed first
* :hpx-pr:`2600` - Compat fixes
* :hpx-pr:`2599` - Compatibility layer for threading support
* :hpx-pr:`2598` - V1.1
* :hpx-pr:`2597` - Release V1.0
* :hpx-pr:`2592` - First attempt to introduce spmd_block in hpx
* :hpx-pr:`2586` - local_segment in segmented_iterator_traits
* :hpx-issue:`2584` - Add allocator support to promise, packaged_task and
  friends
* :hpx-pr:`2576` - Add missing dependencies of cuda based tests
* :hpx-pr:`2575` - Remove warnings due to some captured variables
* :hpx-issue:`2574` - MSVC 2015 Compiler crash when building HPX
* :hpx-issue:`2568` - Remove throttle_scheduler as it has been abandoned
* :hpx-issue:`2566` - Add an inline versioning namespace before 1.0 release
* :hpx-issue:`2565` - Raise minimal cmake version requirement
* :hpx-pr:`2556` - Fixing scan partitioner
* :hpx-pr:`2546` - Broadcast async
* :hpx-issue:`2543` - make install fails due to a non-existing .so file
* :hpx-pr:`2495` - wait_or_add_new returning thread_id_type
* :hpx-issue:`2480` - Unable to register new performance counter
* :hpx-issue:`2471` - no type named 'fcontext_t' in namespace
* :hpx-issue:`2456` - Re-implement hpx::util::unwrapped
* :hpx-issue:`2455` - Add more arithmetic performance counters
* :hpx-pr:`2454` - Fix a couple of warnings and compiler errors
* :hpx-pr:`2453` - Timed executor support
* :hpx-pr:`2447` - Implementing new executor API (P0443)
* :hpx-issue:`2439` - Implement executor proposal
* :hpx-issue:`2408` - Stackoverflow detection for linux, e.g. based on
  libsigsegv
* :hpx-pr:`2377` - Add a customization point for put_parcel so we can override
  actions
* :hpx-issue:`2368` - HPX_ASSERT problem
* :hpx-issue:`2324` - Change default number of threads used to the maximum of
  the system
* :hpx-issue:`2266` - hpx_0.9.99 make tests fail
* :hpx-pr:`2195` - Support for code completion in VIM
* :hpx-issue:`2137` - Hpx does not compile over osx
* :hpx-issue:`2092` - make tests should just build the tests
* :hpx-issue:`2026` - Build HPX with Apple's clang
* :hpx-issue:`1932` - hpx with PBS fails on multiple localities
* :hpx-pr:`1914` - Parallel heap algorithm implementations WIP
* :hpx-issue:`1598` - Disconnecting a locality results in segfault using
  heartbeat example
* :hpx-issue:`1404` - unwrapped doesn't work with movable only types
* :hpx-issue:`1400` - hpx::util::unwrapped doesn't work with non-future types
* :hpx-issue:`1205` - TSS is broken
* :hpx-issue:`1126` - vector<future<T> > does not work gracefully with dataflow,
  when_all and unwrapped
* :hpx-issue:`1056` - Thread manager cleanup
* :hpx-issue:`863` - Futures should not require a default constructor
* :hpx-issue:`856` - Allow runtimemode_connect to be used with security enabled
* :hpx-issue:`726` - Valgrind
* :hpx-issue:`701` - Add RCR performance counter component
* :hpx-issue:`528` - Add support for known failures and warning
  count/comparisons to hpx_run_tests.py

