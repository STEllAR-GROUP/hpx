..
    Copyright (C) 2022      Giannis Gonidelis
    Copyright (C) 2007-2022 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_8_0:

===========================
|hpx| V1.8.0 (May 18, 2022)
===========================

With HPX parallel algorithms been fully adapted to C++20 the new release
achieves full conformance with C++20 concurrency and parallelism facilities. HPX
now supports all of the algorithms as specified by C++20. We have added support
for vectorization to more of our algorithms. Much work has been done towards
implementing |p2300| ("std::execution") and implementing the underlying
senders/receivers facilities. Finally, The new release comes with a brand new
documentation interface!

General changes
===============

- The new documentation can now be found on our webpage: https://hpx-docs.stellar-group.org.
  This includes a completely new and user-friendly interface environment along with
  restructuring of certain components. The content in the "Quick start", "Manual" and
  "Examples" was improved, while the "Build system" page was adapted to include necessary
  information for newcommers.
- With the vectorization support available in modern hardware architectures HPX
  now provides new data-parallel vector execution policies
  ``hpx::execution::simd`` and ``hpx::execution::par_simd`` that enable
  significant speed-up of our parallel algorithm implementations. The following
  algorithms now support SIMD execution:

  - ``copy``, ``copy_n``
  - ``generate``
  - ``adjacent_difference``, ``adjacent_find``
  - ``all_of``, ``any_of``, ``none_of``
  - ``equal``, ``mismatch``,
  - ``inner_product``
  - ``count``, ``count_if``
  - ``fill``, ``fill_n``
  - ``find``, ``find_end``, ``find_first_of``, ``find_if``, ``find_if_not``
  - ``for_each``, ``for_each_n``
  - ``generate``, ``generate_n``.

- Based on top of |p2300| the HPX parallel algorithms now support the pipeline
  syntax towards an effort to unify their usage along with senders/receivers.
  The HPX parallel algorithms can now bind with senders/receivers using the
  pipeline operator. 
- Several changes took place on the executors provided by HPX:
- The executors now support the ``num_cores`` options in order for the user to
  be able to specify the desired number of cores to be used in the correspodning
  execution.
- The ``scheduler`` executor was implemented on top of senders/receivers and can
  be used with all HPX facilities that schedule new work, such as parallel
  algorithms, ``hpx::async``, ``hpx::dataflow``, etc.
- The performance of ``fork_join_executor`` was improved.
- The following algorithms have been added/adapted to be C++20 conformant:

  - ``min_element``
  - ``max_element``
  - ``minmax_element``
  - ``starts_with``
  - ``ends_with``
  - ``swap_ranges``
  - ``unique``
  - ``unique_copy``
  - ``rotate``
  - ``rotate_copy``
  - ``sort``
  - ``shift_left``
  - ``shift_right``
  - ``stable_sort``
  - ``partition``
  - ``partition_copy``
  - ``stable_partition``
  - ``adjacent_difference``
  - ``nth_element``
  - ``partial_sort``
  - ``partial_sort_copy``.

- ``HPX_FORWARD``/``HPX_MOVE`` macros were introduced that replaced the
  ``std::move`` and ``std::forward`` facilities that in the library code.
- Hangs on distributed barrier were fixed.
- The performance of ``scan_partitioner`` was improved.
- Support was added for ``thread_priority`` to the ``parallel_execution_policy`` 
- Regarding senders/receivers and the |p2300| proposal various actions took place.
  ``stop_token`` was adapted to the recent proposal version
  (``in_place_stop_token`` was introduced). Also hint, annotation, priority and
  stacksize properties were added to the scheduler executor. Stop support was
  added to ``when_all``. Support for completion signatures was added. The
  following schedulers and algorithms were added:

  - ``get_completion_scheduler``
  - ``any_sender`` and ``unique_any_sender``
  - ``split`` sender
  - ``transform_mpi`` sender
  - ``transfer`` sender
  - ``let_error``, ``let_stopped``
  - ``get_env`` and related environment queries
  - ``schedule``, ``set_value``, ``set_error``, ``set_done``, ``start`` and
    ``connect`` are now proper customization points as defined in |p2300|.

- Several namespaces were altered towards conformance with C++20. Compatibility layers
  have been added and the old versions will be removed in next releases. The namespace
  changes are the following:

  - ``hpx::parallel::induction/reduction`` were movied into namespace ``hpx::experimental``
  - ``for_loop`` and friends were moved into namespace ``hpx::experimental``.
  - ``hpx::util::optional`` and friends were moved into namespace ``hpx``.
  - ``hpx::lcos::barrier`` has been moved into the ``hpx::distributed`` namespace and
    ``hpx::lcos::local::cpp20_barrier`` has been renamed to ``barrier`` and moved into
    the ``hpx`` namespace.
  - ``hpx::lcos::latch`` has been moved into the ``hpx::distributed`` namespace and
    ``lcos::local::latch`` has been moved into the ``hpx`` namespace. The
    ``count_down_and_wait()`` functionality of ``latch`` has been renamed to
    ``arrive_and_wait()``.
  - ``hpx::util::unique_function_nonser`` has been renamed to ``hpx::move_only_function``.
  - ``hpx::util::unique_function`` has been renamed to ``hpx::distributed::move_only_function``.
  - ``hpx::util::function`` has been renamed to ``hpx::distributed::function``.
  - ``hpx::util::function_nonser`` has been renamed to ``hpx::function``.
  - ``hpx::util::function_ref`` have been moved to namespace ``hpx``.
  - ``hpx::lcos::split_future`` changed namespace and is now used as ``hpx::split_future``.
  - ``hpx::lcos::local::counting_semaphore`` has been deprecated and
    ``hpx::lcos::local::cpp20_counting_semaphore`` has been renamed to
    ``hpx::counting_semaphore``.
  - ``hpx::lcos::local::cpp20_binary_semaphore`` has been renamed to ``hpx::binary_semaphore``.
  - ``hpx::lcos::local::sliding_semaphore`` has been renamed to ``hpx::sliding_semaphore`` and
  - ``hpx::lcos::local::sliding_semaphore_var`` has been renamed to ``hpx::sliding_semaphore_var``.
  - ``hpx::lcos::local::spinlock`` has been renamed to ``hpx::spinlock``.
  - ``hpx::lcos::local::mutex`` has been renamed to ``hpx::mutex``.
  - ``hpx::lcos::local::timed_mutex`` has been renamed to ``hpx::timed_mutex``.
  - ``hpx::lcos::local::no_mutex`` has been renamed to ``hpx::no_mutex``.
  - ``hpx::lcos::local::recursive_mutex`` has been renamed to ``hpx::recursive_mutex``.
  - ``hpx::lcos::local::shared_mutex`` has been renamed to ``hpx::shared_mutex``.
  - ``hpx::lcos::local::upgrade_lock`` has been renamed to ``hpx::upgrade_lock``.
  - ``hpx::lcos::local::upgrade_to_unique_lock`` has been renamed to ``hpx::upgrade_to_unique_lock``.
  - ``hpx::lcos::local::condition_variable`` has been renamed to ``hpx::condition_variable``.
    ``hpx::lcos::local::condition_variable_var`` has been renamed to
    ``hpx::condition_variable_var``.
  - ``hpx::lcos::local::once_flag`` has been renamed to ``hpx::once_flag``, and .
    ``hpx::lcos::local::call_once`` has been renamed to ``hpx::call_once``.

- The new LCI (Lightweight Communication Interface) parcelport was added that supports
  irregular and asynchronous applications like graph analysis, sparce linear algebra,
  modern parallel architectures etc. Major features include:

  - Support for advanced communication primitives like two sided send/recv and
    one sided remote put.
  - Better multi-threaded performance.
  - Explicit user control of communication resource.
  - Flexible signaling mechanisms (synchronizer, completion queue, active message handler). 

- The following CMake flags were added, mostly to support using HPX as a backend
  for SHAD (https://github.com/pnnl/SHAD). Please note that these options enable
  questionable functionalities, partially they even enable undefined behavior.
  Please only use any of them if you know what you're doing:

  - ``HPX_SERIALIZATION_WITH_ALLOW_RAW_POINTER_SERIALIZATION``
  - ``HPX_SERIALIZATION_WITH_ALL_TYPES_ARE_BITWISE_SERIALIZABLE``
  - ``HPX_SERIALIZATION_WITH_ALLOW_CONST_TUPLE_MEMBERS``

Breaking changes
================

- Minimum required C++ standard library is C++17.
- Support for GCC 7 and Clang 8.0.0 and below has been removed.
- CUDA  version required updated to 11.4.
- CMake version required updated to 3.18.
- The default version of Asio used was updated to 1.20.0.
- The default version of APEX used was updated to 2.5.1.
- APEX version was updated to 2.5.1.
- ``tagged_pair`` and ``tagged_tuple`` were removed.
- ``tag_dispatch`` was renamed to ``tag_invoke``.
- ``hpx.max_backgroud_threads`` was renamed to ``hpx.parcel.max_background_threads``.
- The following CMake flags were removed after being deprecated for at least two releases:

  - ``HPX_SCHEDULER_MAX_TERMINATED_THREADS``
  - ``HPX_WITH_GOOGLE_PERFTOOLS``
  - ``HPX_WITH_INIT_START_OVERLOADS_COMPATIBILITY``
  - ``HPX_HAVE_{COROUTINE,PLUGIN}_GCC_HIDDEN_VISIBILITY``
  - ``HPX_TOP_LEVEL``
  - ``HPX_WITH_COMPUTE_CUDA``
  - ``HPX_WITH_ASYNC_CUDA``

- ``annotate_function`` was renamed to ``scoped_annotation``.
- ``execution::transform`` was renamed to ``execution::then``.
- ``execution::detach`` was renamed to ``execution::start_detached``.
- ``execution::on_sender`` was renamed to ``execution::schedule_on``.
- ``execution::just_on`` was renamed to ``execution::just_transfer``.
- ``execution::set_done`` was renamed to ``execution::set_stopped``.

Closed issues
=============

* :hpx-issue:`5871` - distributed::channel.regsiter_as terminates the active task.
* :hpx-issue:`5856` - Performance counters do not compile
* :hpx-issue:`5828` - hpx::distributed:barrier errors
* :hpx-issue:`5812` - OctoTiger does not compile with HPX master and CUDA 11.5
* :hpx-issue:`5784` - HPX failing with co_await and hpx::when_all(futures)
* :hpx-issue:`5774` - CMake can't find HPXCacheVariables.cmake 
* :hpx-issue:`5764` - Fix HIP problem 
* :hpx-issue:`5724` - Missing binary filter compression header
* :hpx-issue:`5721` - Cleanup after repository split
* :hpx-issue:`5701` - It seems that the tcp parcelport is running, and the MPI parcelport is ignored
* :hpx-issue:`5692` - Kokkos compilation fails when using both HPX and CUDA execution spaces with gcc 9.3.0
* :hpx-issue:`5686` - Rename `annotate_function`
* :hpx-issue:`5668` - HPX does not detect the C++ 20 standard using gcc 11.2
* :hpx-issue:`5666` - Compilation error using boost 1.76 and gcc 11.2.1
* :hpx-issue:`5653` - Implement P2248 for our algorithms
* :hpx-issue:`5647` - [User input needed] Remove (CUDA) compute functionality?
* :hpx-issue:`5590` - hello_world_distributed fails on startup with HPX stable, MPICH 3.3.2, on Deep Bayou
* :hpx-issue:`5570` - Rename tag_dispatch to tag_invoke
* :hpx-issue:`5566` - can't build simple example: "Cannot use the dummy implementation of future_then_dispatch"
* :hpx-issue:`5565` - build failure: hpx::string_util::trim()
* :hpx-issue:`5553` - Github action to validate the cff file refs #5471
* :hpx-issue:`5504` - CMake does not work for HPX 1.7.0 on Piz Daint
* :hpx-issue:`5503` - Use contiguous index queue in bulk execution to reduce number of spawned tasks
* :hpx-issue:`5502` - C++20 std::coroutine cmake detection
* :hpx-issue:`5478` - hpx.dll built with vcpkg got functions pointing to the same location
* :hpx-issue:`5472` - Compilation error with cuda/11.3 
* :hpx-issue:`5469` - Compiler warning about HPX_NODISCARD when building with APEX
* :hpx-issue:`5463` - Address minor comments of the C++17 PR bump 
* :hpx-issue:`5456` - Use `std::ranges::iter_swap` where available
* :hpx-issue:`5404` - Build fails with error "Cannot open include file asio/io_context.hpp"
* :hpx-issue:`5381` - Add starts_with and ends_with algorithms
* :hpx-issue:`5344` - Further simplify tag_invoke helpers
* :hpx-issue:`5269` - Allow setting a label on executors/policies
* :hpx-issue:`5219` - (Re-)Implement executor API on top of sender/receiver infrastructure
* :hpx-issue:`5216` - Performance counter module not loading 
* :hpx-issue:`5162` - Require C++17 support
* :hpx-issue:`5156` - Disentangle segmented algorithms
* :hpx-issue:`5118` - Lock held while suspending
* :hpx-issue:`5111` - Tests fail to build with binary_filter plugins enabled
* :hpx-issue:`5110` - Tests don't get built
* :hpx-issue:`5105` - PAPI performance counters not available
* :hpx-issue:`5002` - hpx::lcos::barrier() results in deadlock
* :hpx-issue:`4992` - Clang-format the rest of the files
* :hpx-issue:`4987` - Use std::function in public APIs
* :hpx-issue:`4871` - HEP: conformance to C++20
* :hpx-issue:`4822` - Adapt parallel algorithms to C++20
* :hpx-issue:`4736` - Deprecate hpx::flush and hpx::endl
* :hpx-issue:`4558` -  Prevent work-stealing from stalling
* :hpx-issue:`4495` - Add anchor links to table rows in documentation
* :hpx-issue:`4469` - New thread state: `pending_low`
* :hpx-issue:`4321` - After the modularization the libfabric parcelport does not compile 
* :hpx-issue:`4308` - Using APEX on multinode jobs when HPX_WITH_NETWORKING = OFF
* :hpx-issue:`3995` - Use C++20 std::source_location where available, adapt ours to conform
* :hpx-issue:`3861` - Selected processor does not support 'yield' in ARM mode
* :hpx-issue:`3706` - Add shift_left and shift_right algorithms
* :hpx-issue:`3646` - Parallel algorithms should accept iterator/sentinel pairs
* :hpx-issue:`3636` - HPX Modularization
* :hpx-issue:`3546` - Modularization of HPX
* :hpx-issue:`3474` - Modernize CMake used in HPX
* :hpx-issue:`1836` - hpx::parallel does not have a sort implementation
* :hpx-issue:`1668` - Adapt all parallel algorithms to Ranges TS
* :hpx-issue:`1141` - Implement N4409 on top of HPX

Closed pull requests
====================

* :hpx-pr:`5885` - Testing newer ASIO version
* :hpx-pr:`5884` - Fix miscellaneous doc sections
* :hpx-pr:`5882` - Fixing OctoTiger incompatibility introduced recently
* :hpx-pr:`5881` - Fixing recent patch that disables ATOMIC_FLAG_INIT for C++20 and up
* :hpx-pr:`5880` - refactor: convert `counter_status` enum to enum class
* :hpx-pr:`5878` - Docs: Replaced non-existent create_reducer function with create_communicator
* :hpx-pr:`5877` - Doc updates hpx runtime and resources
* :hpx-pr:`5876` - Updates to documentation; grammar edits.
* :hpx-pr:`5875` - Doc updates starting the hpx runtime
* :hpx-pr:`5874` - Doc updates launching configuring
* :hpx-pr:`5873` - Prevent certain generated files from being deleted on reconfigure
* :hpx-pr:`5870` - Adding support for the PJM batch environment
* :hpx-pr:`5867` - Update CMakeLists.txt
* :hpx-pr:`5866` - add cmake option HPX_WITH_PARCELPORT_COUNTERS
* :hpx-pr:`5864` - ATOMIC_INIT_FLAG is deprecated starting C++20
* :hpx-pr:`5863` - Adding llvm 14.0.0 with boost 1.79.0 to Jenkins
* :hpx-pr:`5861` - Let install step proceed on CircleCI even if the segmented algorithms fail
* :hpx-pr:`5860` - Updating APEX tag
* :hpx-pr:`5859` - Splitting documentation generation steps on CircleCI
* :hpx-pr:`5854` - Fixing left-overs from changing counter_type to enum class
* :hpx-pr:`5853` - Adding HPX dependency tool (adapted from Boostdep tool)
* :hpx-pr:`5852` - Optimize LCI parcelport
* :hpx-pr:`5851` - Forking dynamic_bitset from Boost
* :hpx-pr:`5850` - Convert perf_counters::counter_type enum to enum class.
* :hpx-pr:`5849` - Update LCI parcelport to LCI v1.7.1
* :hpx-pr:`5848` - Fedora related fixes
* :hpx-pr:`5847` - Fix API, troubleshooting & people
* :hpx-pr:`5844` - Attempting to fix timeouts of segmented iterator tests
* :hpx-pr:`5842` - change the default value of HPX_WITH_LCI_TAG to v1.7
* :hpx-pr:`5841` - Move the split_future facilities into the namespace hpx
* :hpx-pr:`5840` - wait_xxx_nothrow functions return whether one of the futures is exceptional
* :hpx-pr:`5839` - Moving a list of synchronization primitives into namespace hpx 
* :hpx-pr:`5837` - Moving latch types to hpx and hpx::distributed namespaces
* :hpx-pr:`5835` - Add missing compatibility layer for id_type::management_type values
* :hpx-pr:`5834` - API docs changes 
* :hpx-pr:`5831` - Further improvement actions to rotate
* :hpx-pr:`5830` - Exposing zero-copy serialization threshold through configuration option
* :hpx-pr:`5829` - Attempting to fix failing barrier test
* :hpx-pr:`5827` - Add back explicit template parameter to `ignore_while_checking` to compile with nvcc
* :hpx-pr:`5826` - Reduce number of allocations while calling async_bulk_execute
* :hpx-pr:`5825` - Steal from neighboring NUMA domain only 
* :hpx-pr:`5823` - Remove obsolete directories and adjust build system
* :hpx-pr:`5822` - Clang-format remaining files 
* :hpx-pr:`5821` - Enable permissive- flag on Windows GitHub actions builders
* :hpx-pr:`5820` - Convert throwmode enum to enum class
* :hpx-pr:`5819` - Marking customization points for intrusive_ptr as noexcept
* :hpx-pr:`5818` - Unconditionally use C++17 attributes
* :hpx-pr:`5817` - Modernize naming modules
* :hpx-pr:`5816` - Modernize cache module
* :hpx-pr:`5815` - Reapply flyby changes from #5467
* :hpx-pr:`5814` - Avoid test timeouts by reducing test sizes
* :hpx-pr:`5813` - The CUDA problem is not fixed in V11.5 yet...
* :hpx-pr:`5811` - Make sure reduction value is properly moved, when possible
* :hpx-pr:`5810` - Improve error reporting during device initialization in HIP environments
* :hpx-pr:`5809` - Converting scheduler enums into enum class
* :hpx-pr:`5808` - Deprecate hpx::flush and friends
* :hpx-pr:`5807` - Use C++20 std::source_location, if available
* :hpx-pr:`5806` - Moving promise and packaged_task to new namespaces
* :hpx-pr:`5805` - Attempting to fix a test failure when using the LCI parcelpor
* :hpx-pr:`5803` - Attempt to fix CUDA related OctoTiger problems
* :hpx-pr:`5800` - Add option to restrict MPI background work to subset of cores
* :hpx-pr:`5798` - Adding MPI as a dependency to APEX
* :hpx-pr:`5797` - Extend Sphinx role to support arbitrary text to display on a link
* :hpx-pr:`5796` - Disable CUDA tests that cause NVCC to silently fail without error messages
* :hpx-pr:`5795` - Avoid writing path and directories into HPXCacheVariables.cmake
* :hpx-pr:`5793` - Remove features that are deprecated since V1.6
* :hpx-pr:`5792` - Making sure num_cores is properly handled by parallel_executor
* :hpx-pr:`5791` - Moving bind, bind_front, bind_back to namespace hpx
* :hpx-pr:`5790` - Moving serializable function/move_only_function into namespace hpx::distributed
* :hpx-pr:`5787` - Remove unneeded (and commented) tests
* :hpx-pr:`5786` - Attempting to fix hangs in distributed barrier
* :hpx-pr:`5785` - add cmake code to detect arm64 on macOS
* :hpx-pr:`5783` - Moving function and function_ref into namespace hpx
* :hpx-pr:`5781` - Updating used version of Visual Studio
* :hpx-pr:`5780` - Update Piz Daint Jenkins configurations from gcc/clang 7 to 8
* :hpx-pr:`5778` - Updated for_loop.hpp
* :hpx-pr:`5777` - Update reference for foreach benchmark
* :hpx-pr:`5775` - Move optional into namespace hpx
* :hpx-pr:`5773` - Moving barrier to consolidated namespaces
* :hpx-pr:`5772` - Adding missing docs for ranges::find_if and find_if_not algorithms
* :hpx-pr:`5771` - Moving for_loop into namespace hpx::experimental
* :hpx-pr:`5770` - Fixing HIP issues
* :hpx-pr:`5769` - Slight improvement of small_vector performance
* :hpx-pr:`5766` - Fixing a integral conversion warning
* :hpx-pr:`5765` - Adding a sphinx role allowing to link to a file directly in github
* :hpx-pr:`5763` - add num_cores facility
* :hpx-pr:`5762` - Fix Public API main page
* :hpx-pr:`5761` - Add missing inline to mpi_exception.hpp error_message function
* :hpx-pr:`5760` - Update cdash build url
* :hpx-pr:`5759` - Switch to use generic rostam SLURM partitions
* :hpx-pr:`5758` - Adding support for P2300 completion signatures
* :hpx-pr:`5757` - Fix missing links in Public API 
* :hpx-pr:`5756` - Add stop support to when_all
* :hpx-pr:`5755` - Support for data-parallelism for mismatch algorithm
* :hpx-pr:`5754` - Support for data-parallelism for equal algorithm
* :hpx-pr:`5751` - Propagate MPI dependencies to command line handling
* :hpx-pr:`5750` - Make sure required MPI initialization flags are properly applied and supported
* :hpx-pr:`5749` - P2300 stop token
* :hpx-pr:`5748` - Adding environmental query CPOs
* :hpx-pr:`5747` - Renaming set_done to set_stopped (as per P2300)
* :hpx-pr:`5745` - Modernize serialization module
* :hpx-pr:`5743` - Add check for MPICH and set the correct env to support multi-threaded
* :hpx-pr:`5742` - Remove obsolete files related to cpuid, etc.
* :hpx-pr:`5741` - Support for data-parallelism for adjacent find
* :hpx-pr:`5740` - Support for data-parallelism for find algorithms
* :hpx-pr:`5739` - Enable the option to attach a debugger on a segmentation fault (linux)
* :hpx-pr:`5738` - Fixing spell-checking errors
* :hpx-pr:`5737` - Attempt to fix migrate_component issue
* :hpx-pr:`5736` - Set commit status from Jenkins also for special branches 
* :hpx-pr:`5734` - Revert #5586
* :hpx-pr:`5732` - Attempt to improve build-id reporting to cdash
* :hpx-pr:`5731` - Randomly delay execution of bash scripts launched by Jenkins
* :hpx-pr:`5729` - Workaround for CMake/Ninja generator OOM problem
* :hpx-pr:`5727` - Moving compression plugins to components directory
* :hpx-pr:`5726` - Moving/consolidating parcel coalescing plugin sources
* :hpx-pr:`5725` - Making sure headers for serialization filters are being installed
* :hpx-pr:`5723` - Moving more tests to modules
* :hpx-pr:`5722` - Removing superfluous semicolons
* :hpx-pr:`5720` - Moving parcelports into modules
* :hpx-pr:`5719` - Moving more files to parcelset module
* :hpx-pr:`5718` - build: refactor sphinx config file 
* :hpx-pr:`5717` - Creating parcelset modules
* :hpx-pr:`5716` - Avoid duplicate definition error
* :hpx-pr:`5715` - The new LCI parcelport for HPX
* :hpx-pr:`5714` - Refine propagation of HPX_WITH_... options
* :hpx-pr:`5713` - Significantly reduce CI jobs run on Piz Daint
* :hpx-pr:`5712` - Updating jenkins configuration for Rostam2.2
* :hpx-pr:`5711` - Refactor manual sections
* :hpx-pr:`5710` - Making task_group serializable
* :hpx-pr:`5709` - Update the MPI cmake setup
* :hpx-pr:`5707` - Better diagnose parcel bootstrap problems
* :hpx-pr:`5704` - Test with hwloc 2.7.0 with GCC 11
* :hpx-pr:`5703` - Fix `counting_iterator` container tests
* :hpx-pr:`5702` - Attempting to fix CircleCI timeouts
* :hpx-pr:`5699` - Update CI to use Boost 1.78.0
* :hpx-pr:`5697` - Adding fork_join_executor to foreach_benchmark
* :hpx-pr:`5696` - Modernize when_all and friends (when_any, when_some, when_each)
* :hpx-pr:`5693` - Fix test errors with `_GLIBCXX_DEBUG` defined
* :hpx-pr:`5691` - Rename `annotate_function` to `scoped_annotation`
* :hpx-pr:`5690` - Replace tag_dispatch with tag_invoke in minmax segmented
* :hpx-pr:`5688` - Remove more deprecated macros
* :hpx-pr:`5687` - Add most important CMake options
* :hpx-pr:`5685` - Fix future API
* :hpx-pr:`5684` - Move lock registration to separate module and remove global lock registration
* :hpx-pr:`5683` - Make hpx::wait_all etc. throw exceptions when waited futures hold exceptions and deprecate hpx::lcos::wait_all[_n] in favor of hpx::wait_all[_n]
* :hpx-pr:`5682` - Fix macOS test exceptions
* :hpx-pr:`5681` - docs: add links to hpx recepies
* :hpx-pr:`5680` - Embed base execution policies to datapar execution policies
* :hpx-pr:`5679` - Fix `fork_join_executor` with dynamic schedule
* :hpx-pr:`5678` - Fix compilation of service executors with nvcc
* :hpx-pr:`5677` - Remove compute_cuda module
* :hpx-pr:`5676` - Don't require up-to-date approvals for bors
* :hpx-pr:`5675` - Add default template type parameters for algorithms
* :hpx-pr:`5674` - Allow using  `any_sender` in global variables
* :hpx-pr:`5671` - Making sure task_group can be reused
* :hpx-pr:`5670` - Relax constraints on `execution::when_all`
* :hpx-pr:`5669` - Use HPX_WITH_CXX_STANDARD for controlling C++ version 
* :hpx-pr:`5667` - Attempt to fix compilation issues with Boost V1.76
* :hpx-pr:`5664` - Change logging errors to warnings in schedulers
* :hpx-pr:`5663` - Use dynamic bitsets by default for CPU masks
* :hpx-pr:`5662` - Disambiguate namespace for MSVC
* :hpx-pr:`5660` - Replacing remaining std::forward and std::move with HPX_FORWARD and HPX_MOVE
* :hpx-pr:`5659` - Modernize hpx::future and related facilities
* :hpx-pr:`5658` - Replace HPX_INLINE_CONSTEXPR_VARIABLE with inline constexpr
* :hpx-pr:`5657` - Remove tagged, tagged_pair and tagged_tuple, remove tuple/pair specializations
* :hpx-pr:`5656` - Rename on execution::schedule_from, rename just_on to just_transfer, and add transfer
* :hpx-pr:`5655` - Avoid for module lists to grow indefinitely in cmake cache
* :hpx-pr:`5649` - build: replace usage of Python's reserved words and functions as variable names
* :hpx-pr:`5648` - Modernize action modules and related code
* :hpx-pr:`5646` - Fix ends_with test
* :hpx-pr:`5645` - Add matrix multiplication example
* :hpx-pr:`5644` - Rename execution::transform to execution::then and execution::detach to execution::start_detached
* :hpx-pr:`5643` - Update performance test references
* :hpx-pr:`5642` - Adapting adjacent_difference to work with proxy iterators
* :hpx-pr:`5641` - Factorize perftests scripts
* :hpx-pr:`5640` - Fixed links to sources in Sphinx documentation
* :hpx-pr:`5639` - Fix generate datapar tests for Vc
* :hpx-pr:`5638` - Simd all any none
* :hpx-pr:`5637` - Use bors for merging pull requests
* :hpx-pr:`5636` - Fix leftover std::holds_alternative usage
* :hpx-pr:`5635` - Update container image tag in GitHub actions HIP configuration
* :hpx-pr:`5633` - Moving packaged_task to module futures
* :hpx-pr:`5632` - Tell Asio to use std::aligned_new only if available
* :hpx-pr:`5631` - Adding tag parameter to channel communicator get/set
* :hpx-pr:`5630` - Add partial_sort_copy and adapt partial sort to c++ 20
* :hpx-pr:`5629` - Set HPX_WITH_FETCH_ASIO to OFF as available in the docker image
* :hpx-pr:`5628` - Add Clang 13 CI configuration
* :hpx-pr:`5627` - Replace alternative keyword
* :hpx-pr:`5626` - docs: add support for BibTeX references in Sphinx docs
* :hpx-pr:`5624` - Fix pkgconfig replacements involving CMAKE_INSTALL_PREFIX
* :hpx-pr:`5623` - build: remove unused import from conf.py.in
* :hpx-pr:`5622` - Remove HPX_WITH_VCPKG CMake option
* :hpx-pr:`5621` - Replacing boost::container::small_vector
* :hpx-pr:`5620` - Update Asio tag from 1.18.2 to 1.20.0
* :hpx-pr:`5619` - Fix block_os_threads_1036 test
* :hpx-pr:`5618` - Make sure condition variables are notified under a lock in the thread_pool_scheduler test
* :hpx-pr:`5617` - Use advance_and_get_distance where required
* :hpx-pr:`5616` - Remove separately building segmented algorithms on CircleCI
* :hpx-pr:`5613` - Fix Vc datapar adjacent_difference
* :hpx-pr:`5609` - docs: add anchor links to performance counter tables
* :hpx-pr:`5608` - Fix header test error by adding missing numeric
* :hpx-pr:`5607` - Fix simd adj diff
* :hpx-pr:`5605` - Fix usage of HPX_INVOKE macro
* :hpx-pr:`5604` - Make use of shell-session to allow non-copyable $
* :hpx-pr:`5603` - Suppress some MSVC warnings in C++20 mode
* :hpx-pr:`5602` - Test HPX_DATASTRUCTURES_WITH_ADAPT_STD_TUPLE=OFF to one CI configuration
* :hpx-pr:`5601` - Test case for any_sender should use hpx::tuple
* :hpx-pr:`5600` - Rename tag_dispatch back to tag_invoke
* :hpx-pr:`5599` - Change theme, fix Quickstart & Examples
* :hpx-pr:`5596` - Use precompiled headers in tests
* :hpx-pr:`5595` - Drop semicolons for macro calls
* :hpx-pr:`5594` - Adapt datapar generate
* :hpx-pr:`5593` - Update any_sender to use tag_dispatch for execution customizations
* :hpx-pr:`5592` - Add nth_element
* :hpx-pr:`5591` - Remove unnecessary checks for C++17 for tests
* :hpx-pr:`5589` - Add HPX_FORWARD/HPX_MOVE macros
* :hpx-pr:`5588` - Fixing the output formatting for id_types
* :hpx-pr:`5586` - Remove local functionality
* :hpx-pr:`5585` - Delete GitExternal.cmake
* :hpx-pr:`5584` - Serialization of hpx::tuple must use hpx::get
* :hpx-pr:`5583` - fix coroutine_traits allocate calls, add unhandled_exception() implementation.
* :hpx-pr:`5582` - Make more examples work with local runtime
* :hpx-pr:`5581` - Add support for several performance tests in CI
* :hpx-pr:`5580` - Adapt simd adj diff
* :hpx-pr:`5579` - Split absolute paths for generated pkg-config files into -L/-l parts
* :hpx-pr:`5577` - fix unit fill test for datapar with Vc
* :hpx-pr:`5576` - Update forgotten "Full" names
* :hpx-pr:`5575` - Change scan partitioner implementation
* :hpx-pr:`5574` - Remove a few deprecated and unused CMake options
* :hpx-pr:`5572` - Remove more guards for the distributed runtime
* :hpx-pr:`5571` - Add workaround for libstc++ in string_util trim
* :hpx-pr:`5569` - Use no_unique_address in sender adaptors
* :hpx-pr:`5568` - Change try catch block to try_catch_exception_ptr
* :hpx-pr:`5567` - Make default_agent::yield actually yield
* :hpx-pr:`5564` - Adjacent
* :hpx-pr:`5562` - More changes to overcome build problems on Windows after recent module rearrangements
* :hpx-pr:`5560` - Update tests and examples
* :hpx-pr:`5559` - Fixing cmake folder names after module restructuring
* :hpx-pr:`5558` - Fixing wrong module dependencies
* :hpx-pr:`5557` - Adding an example for the new channel_communicator API
* :hpx-pr:`5556` - Remove leftover thread pool os executor tests
* :hpx-pr:`5555` - Add option enabling serializing raw pointers
* :hpx-pr:`5554` - Make sure command line aliasing is properly handled
* :hpx-pr:`5552` - Modernizing some of the async facilities
* :hpx-pr:`5551` - Fixing for local executions of actions to properly set task names
* :hpx-pr:`5550` - Update CUDA module in clang-cuda configuration
* :hpx-pr:`5549` - Fixing agent_ref::yield_k to actually call yield_k
* :hpx-pr:`5548` - Making get_action_name() noexcept
* :hpx-pr:`5547` - Fixing communication set
* :hpx-pr:`5546` - Fixing shutdown problems caused by missing ref-counting
* :hpx-pr:`5545` - Remove wrong move in thread_pool_scheduler_bulk.hpp
* :hpx-pr:`5543` - Extend launch policy to carry stack size and scheduling hint in addition to priority
* :hpx-pr:`5542` - Simplify execution CPOs
* :hpx-pr:`5540` - Adapt partition, partition_copy and stable_partition to C++ 20
* :hpx-pr:`5539` - Adapt mismatch to support sentinels
* :hpx-pr:`5538` - Document specific sphinx version required for the documentation
* :hpx-pr:`5537` - Test release and debug builds on Piz Daint
* :hpx-pr:`5536` - This fixes referencing stale iterators during the execution of binary mismatch
* :hpx-pr:`5535` - Rename simdpar to par_simd
* :hpx-pr:`5534` - Fix Quick start & Manual Docs
* :hpx-pr:`5533` - Fix `annotate_function` for `std::string`
* :hpx-pr:`5532` - Update two remaining apex links from khuck to UO-OACISS
* :hpx-pr:`5531` - Use contiguous_index_queue in thread_pool_scheduler
* :hpx-pr:`5530` - Eagerly initialize a configurable number of threads on scheduler/thread queue init
* :hpx-pr:`5529` - Update benchmarks and add support for scheduler_executor
* :hpx-pr:`5528` - Add missing properties to executors/schedulers
* :hpx-pr:`5527` - Set local thread/pool number in local/static_queue_scheduler
* :hpx-pr:`5526` - Update Rostam HIP configuration to use 4.3.0
* :hpx-pr:`5525` - Fix Building HPX in Quick start
* :hpx-pr:`5524` - Upload image on cdash
* :hpx-pr:`5523` - Modernize facilities related to hpx::sync
* :hpx-pr:`5522` - Add sender overloads for remaining algorithms
* :hpx-pr:`5521` - Minor changes that improve performance
* :hpx-pr:`5520` - Update reference as perftests failing regularly
* :hpx-pr:`5519` - Add transform_mpi sender adapter
* :hpx-pr:`5518` - Add sender overloads to rotate/rotate_copy
* :hpx-pr:`5517` - Fix coroutine integration
* :hpx-pr:`5515` - Avoid deadlock in ignore_while_locked_1485 test
* :hpx-pr:`5514` - Add split sender adapter
* :hpx-pr:`5512` - Update Rostam HIP configuration
* :hpx-pr:`5511` - Fix Asio target name for precompiled headers
* :hpx-pr:`5510` - Add any_sender and unique_any_sender
* :hpx-pr:`5509` - Test with Boost 1.77 on gcc/clang-newest configurations
* :hpx-pr:`5508` - Minor release changes from 1.7.1
* :hpx-pr:`5507` - Add missing commits from scheduler_executor PR
* :hpx-pr:`5506` - Fix condition for checking if we should use our own variant
* :hpx-pr:`5501` - Attempt to fix thread_pool_scheduler test
* :hpx-pr:`5493` - Update Jenkins GitHub token to use StellarBot GitHub account
* :hpx-pr:`5490` - Fix clang-format error on master
* :hpx-pr:`5487` - Add get_completion_scheduler CPO and customize bulk for thread_pool_scheduler
* :hpx-pr:`5484` - Add missing header to jacobi_component/server/solver.hpp
* :hpx-pr:`5481` - Changing the APEX repository to the new location
* :hpx-pr:`5479` - Fix version check for CUDA noexcept/result_of bug
* :hpx-pr:`5477` - Require cxx17 minor comments
* :hpx-pr:`5476` - Fix cmake format error
* :hpx-pr:`5475` - Require CMake 3.18 as it is already a requirement for CUDA
* :hpx-pr:`5474` - Make the cuda parameters of try_compile optional
* :hpx-pr:`5473` - Update cuda arch and change cuda version
* :hpx-pr:`5471` - Add corrected citation.cff
* :hpx-pr:`5470` - Adapt stable_sort to C++ 20
* :hpx-pr:`5468` - Experimentation to make the perftest report public
* :hpx-pr:`5466` - Add shift_left and shift_right algorithms
* :hpx-pr:`5465` - Adapt datapar fill
* :hpx-pr:`5464` - Moving tag_dispatch to separate module
* :hpx-pr:`5461` - Rename HPX_WITH_CUDA_COMPUTE with HPX_WITH_COMPUTE_CUDA
* :hpx-pr:`5460` - Adapt sort to C++ 20
* :hpx-pr:`5459` - Adapt rotate/rotate_copy to C++20
* :hpx-pr:`5458` - Adapt unique and unique_copy to C++ 20
* :hpx-pr:`5455` - Remove and clean up fallback sender implementations
* :hpx-pr:`5454` - Make performance plot show even if similar performance
* :hpx-pr:`5453` - Post 1.7.0 version bump
* :hpx-pr:`5452` - Fix find_end parallel overload
* :hpx-pr:`5450` - Change the print-bind output to be more precise
* :hpx-pr:`5449` - Adapt swap_ranges to C++ 20
* :hpx-pr:`5446` - Use more verbose names in sender algorithms
* :hpx-pr:`5443` - Properly support ASAN with MSVC
* :hpx-pr:`5441` - Adding reference counting to thread_data
* :hpx-pr:`5429` - Scheduler executor
* :hpx-pr:`5428` - Adapt datapar copy
* :hpx-pr:`5421` - Update CI base image to use clang-format 11
* :hpx-pr:`5410` - Add ranges starts_with and ends_with algorithms
* :hpx-pr:`5383` - Tentatively remove runtime_registration_wrapper from cuda futures
* :hpx-pr:`5377` - Fewer Asio includes and more precompiled headers
* :hpx-pr:`5329` - Sender overloads for parallel algorithms
* :hpx-pr:`5313` - Rearrange modules between libraries
* :hpx-pr:`5283` - Require minimum C++17 and change CUDA handling
* :hpx-pr:`5241` - Adapt min_element, max_element and minmax_element to C++20
