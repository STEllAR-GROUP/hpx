..
    Copyright (C) 2007-2020 Hartmut Kaiser
    Copyright (C)      2020 ETH Zurich

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_5_0:

===========================
|hpx| V1.5.0 (Sep 02, 2020)
===========================

General changes
===============

The main focus of this release is on APIs and C++20 conformance. We have added
many new C++20 features and adapted multiple algorithms to be fully C++20
conformant. As part of the modularization we have begun specifying the public
API of |hpx| in terms of headers and functionality, and aligning it more closely
to the C++ standard. All non-distributed modules are now in place, along with an
experimental option to completely disable distributed features in |hpx|. We have
also added experimental asynchronous MPI and CUDA executors. Lastly this release
introduces |cmake| targets for depending projects, performance improvements,
and many bug fixes.

* We have added the C++20 features ``hpx::jthread`` and ``hpx::stop_token``.
  ``hpx::condition_variable_any`` now exposes new functions supporting
  ``hpx::stop_token``.
* We have added ``hpx::stable_sort`` based on Francisco Tapia's
  implementation.
* We have adapted existing synchronization primitives to be fully conformant
  C++20: ``hpx::barrier``, ``hpx::latch``, ``hpx::counting_semaphore``, and
  ``hpx::binary_semaphore``.
* We have started using customization point objects (CPOs) to make the
  corresponding algorithms fully conformant to C++20 as well as to make
  algorithm extension easier for the user. ``all_of``/``any_of``/``none_of``,
  ``copy``, ``count``, ``destroy``, ``equal``, ``fill``, ``find``, ``for_each``,
  ``generate``, ``mismatch``, ``move``, ``reduce``, ``transform_reduce`` are
  using those CPOs (all in namespace ``hpx``).  We also have adapted their
  corresponding ``hpx::ranges`` versions to be conforming to C++20 in this
  release.
* We have adapted support for ``co_await`` to C++20, in addition to
  ``hpx::future`` it now also supports ``hpx::shared_future``. We have also
  added allocator support for futures returned by ``co_return``. It is no longer
  in the ``experimental`` namespace.
* We added serialization support for ``std::variant`` and ``std::tuple``.
* ``result_of`` and ``is_callable`` are now deprecated and replaced by
  ``invoke_result`` and ``is_invocable`` to conform to C++20.
* We continued with the modularization, making it easier for us to add the new
  experimental ``HPX_WITH_DISTRIBUTED_RUNTIME`` |cmake| option (see below) . An
  significant amount of headers have been deprecated. We adapted the namespaces
  and headers we could to be closer to the standard ones (:ref:`public_api`).
  Depending code should still compile, however warnings are now generated
  instructing to change the include statements accordingly.
* It is now possible to have a basic CUDA support including a helper function to
  get a future from a CUDA stream and target handling. They are available under
  the ``hpx::cuda::experimental`` namespace and they can be enabled with the
  ``-DHPX_WITH_ASYNC_CUDA=ON`` |cmake| option.
* We added a new ``hpx::mpi::experimental`` namespace for getting futures from
  an asynchronous MPI call and a new minimal MPI executor
  ``hpx::mpi::experimental::executor``. These can be enabled with the
  ``-DHPX_WITH_ASYNC_MPI=On`` |cmake| option.
* A polymorphic executor has been implemented to reduce compile times as a
  function accepting executors can potentially be instantiated only once instead
  of multiple times with different executors. It accepts the function signature
  as a template argument. It needs to be constructed from any other executor.
  Please note, that the function signatures that can be scheduled using
  ``then_execute``, ``bulk_sync_execute``, ``bulk_async_execute`` and
  ``bulk_then_execute`` are slightly different (See the comment in
  :hpx-pr:`4514` for more details).
* The underlying executor of ``block_executor`` has been updated to a newer one.
* We have added a parameter to ``auto_chunk_size`` to control the amount of
  iterations to measure.
* All executor parameter hooks can now be exposed through the executor itself.
  This will allow to deprecate the ``.with()`` functionality on execution
  policies in the future. This is also a first step towards simplifying our
  executor APIs in preparation for the upcoming C++23 executors
  (senders/receivers).
* We have moved all of the existing APIs related to resiliency into the
  namespace ``hpx::resiliency::experimental``. Please note this is a breaking
  change without backwards-compatibility option. We have converted all of those
  APIs to be based on customization point objects. Two new executors have been
  added to enable easy integration of the existing resiliency features with
  other facilities (like the parallel algorithms): ``replay_executor`` and
  ``replicate_executor``.
* We have added performance counters type information (``aggregating``,
  ``monotonically increasing``, ``average count``, ``average timer``, etc.).
* HPX threads are now re-scheduled on the same worker thread they were suspended
  on to avoid cache misses from moving from one thread to the other. This
  behavior doesn't prevent the thread from being stolen, however.
* We have added a new configuration option ``hpx.exception_verbosity`` to allow
  to control the level of verbosity of the exceptions (3 levels available).
* ``broadcast_to``, ``broadcast_from``, ``scatter_to`` and ``scatter_from`` have
  been added to the collectives, modernization of ``gather_here`` and
  ``gather_there`` with futures taken by rvalue references. See the breaking
  change on ``all_to_all`` in the next section. None of the collectives need
  supporting macros anymore (e.g. specifying the data types used for a
  collective operation using ``HPX_REGISTER_ALLGATHER`` and similar is not
  needed anymore).
* New API functions have been added: a) to get the number of cores which are idle
  (``hpx::get_idle_core_count``) and b) returning a bitmask
  representing the currently idle cores (``hpx::get_idle_core_mask``).
* We have added an experimental option to only enable the local runtime, you can
  disable the distributed runtime with ``HPX_WITH_DISTRIBUTED_RUNTIME=OFF``. You
  can also enable the local runtime by using the ``--hpx:local`` runtime option.
* We fixed task annotations for actions.
* The alias ``hpx::promise`` to ``hpx::lcos::promise`` is now deprecated. You
  can use ``hpx::lcos::promise`` directly instead. ``hpx::promise`` will refer
  to the local-only promise in the future.
* We have added a ``prepare_checkpoint`` API function that calculates the
  amount of necessary buffer space for a particular set of arguments
  checkpointed.
* We have added ``hpx::upgrade_lock`` and ``hpx::upgrade_to_unique_lock``, which
  make ``hpx::shared_mutex`` (and similar) usable in more flexible ways.
* We have changed the |cmake| targets exposed to the user, it now includes
  ``HPX::hpx``, ``HPX::wrap_main`` (``int main`` as the first |hpx| thread of
  the application, see :ref:`starting_hpx`),
  ``HPX::plugin``, ``HPX::component``.  The |cmake| variables
  ``HPX_INCLUDE_DIRS`` and ``HPX_LIBRARIES`` are deprecated and will be removed
  in a future release, you should now link directly to the ``HPX::hpx`` |cmake|
  target.
* A new example is demonstrating how to create and use a wrapping executor
  (``quickstart/executor_with_thread_hooks.cpp``)
* A new example is demonstrating how to disable thread stealing during the
  execution of parallel algorithms
  (``quickstart/disable_thread_stealing_executor.cpp``)
* We now require for our |cmake| build system configuration files to be
  formatted using cmake-format.
* We have removed more dependencies on various Boost libraries.
* We have added an experimental option enabling unity builds of HPX using the
  ``-DHPX_WITH_UNITY_BUILD=On`` |cmake| option.
* Many bug fixes.

Breaking changes
================

* |hpx| now requires a C++14 capable compiler. We have set the |hpx| C++
  standard automatically to C++14 and if it needs to be set explicitly, it
  should be specified through the ``CMAKE_CXX_STANDARD`` setting as mandated
  by |cmake|. The ``HPX_WITH_CXX*`` variables are now deprecated and will be
  removed in the future.
* Building and using HPX is now supported only when using |cmake| V3.13 or later,
  Boost V1.64 or newer, and when compiling with clang V5, gcc V7, or VS2019, or
  later. Other compilers might still work but have not been tested thoroughly.
* We have added a ``hpx::init_params`` struct to pass parameters for |hpx|
  initialization e.g. the resource partitioner callback to initialize thread
  pools (:ref:`using_resource_partitioner`).
* The ``all_to_all`` algorithm is renamed to ``all_gather``, and the new
  ``all_to_all`` algorithm is not compatible with the old one.
* We have moved all of the existing APIs related to resiliency into the
  namespace ``hpx::resiliency::experimental``.

Closed issues
=============

* :hpx-issue:`4918` - Rename distributed_executors module
* :hpx-issue:`4900` - Adding JOSS status badge to README
* :hpx-issue:`4897` - Compiler warning, deprecated header used by HPX itself
* :hpx-issue:`4886` - A future bound to an action executing on a different locality doesn't capture exception state
* :hpx-issue:`4880` - Undefined reference to main build error when HPX_WITH_DYNAMIC_HPX_MAIN=OFF
* :hpx-issue:`4877` - hpx_main might not able to start hpx runtime properly
* :hpx-issue:`4850` - Issues creating templated component
* :hpx-issue:`4829` - Spack package & HPX_WITH_GENERIC_CONTEXT_COROUTINES
* :hpx-issue:`4820` - PAPI counters don't work
* :hpx-issue:`4818` - HPX can't be used with IO pool turned off
* :hpx-issue:`4816` - Build of HPX fails when find_package(Boost) is called before FetchContent_MakeAvailable(hpx)
* :hpx-issue:`4813` - HPX MPI Future failed
* :hpx-issue:`4811` - Remove HPX::hpx_no_wrap_main target before 1.5.0 release
* :hpx-issue:`4810` - In hpx::for_each::invoke_projected the hpx::util::decay is misguided
* :hpx-issue:`4787` - `transform_inclusive_scan` gives incorrect results for non-commutative operator
* :hpx-issue:`4786` - transform_inclusive_scan tries to implicitly convert between types, instead of using the provided `conv` function
* :hpx-issue:`4779` - HPX build error with GCC 10.1
* :hpx-issue:`4766` - Move HPX.Compute functionality to experimental namespace
* :hpx-issue:`4763` - License file name
* :hpx-issue:`4758` - CMake profiling results
* :hpx-issue:`4755` - Building HPX with support for PAPI fails
* :hpx-issue:`4754` - CMake cache creation breaks when using HPX with mimalloc
* :hpx-issue:`4752` - HPX MPI Future build failed
* :hpx-issue:`4746` - Memory leak when using dataflow icw components
* :hpx-issue:`4731` - Bug in stencil example, calculation of locality IDs
* :hpx-issue:`4723` - Build fail with NETWORKING OFF
* :hpx-issue:`4720` - Add compatibility headers for modules that had their module headers implicitly generated in 1.4.1
* :hpx-issue:`4719` - Undeprecate some module headers
* :hpx-issue:`4712` - Rename HPX_MPI_WITH_FUTURES option
* :hpx-issue:`4709` - Make deprecation warnings overridable in dependent projects
* :hpx-issue:`4691` - Suggestion to fix and enhance the thread_mapper API
* :hpx-issue:`4686` - Fix tutorials examples
* :hpx-issue:`4685` - HPX distributed map fails to compile
* :hpx-issue:`4680` - Build error with HPX_WITH_DYNAMIC_HPX_MAIN=OFF
* :hpx-issue:`4679` - Build error for hpx w/ Apex on Summit
* :hpx-issue:`4675` - build error with HPX_WITH_NETWORKING=OFF
* :hpx-issue:`4674` - Error running Quickstart tests on OS X
* :hpx-issue:`4662` - MPI initialization broken when networking off
* :hpx-issue:`4652` - How to fix distributed action annotation
* :hpx-issue:`4650` - thread descriptions are broken...again
* :hpx-issue:`4648` - Thread stacksize not properly set
* :hpx-issue:`4647` - Rename generated collective headers in modules
* :hpx-issue:`4639` - Update deprecation warnings in compatibility headers to point to collective headers
* :hpx-issue:`4628` - mpi parcelport totally broken
* :hpx-issue:`4619` - Fully document hpx_wrap behaviour and targets
* :hpx-issue:`4612` - Compilation issue with HPX 1.4.1 and 1.4.0
* :hpx-issue:`4594` - Rename modules
* :hpx-issue:`4578` - Default value for HPX_WITH_THREAD_BACKTRACE_DEPTH
* :hpx-issue:`4572` - Thread manager should be given a runtime_configuration
* :hpx-issue:`4571` - Add high-level documentation to new modules
* :hpx-issue:`4569` - Annoying warning when compiling - pls suppress or fix it.
* :hpx-issue:`4555` - HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION compilation error
* :hpx-issue:`4543` - Segfaults in Release builds using `sleep_for`
* :hpx-issue:`4539` - Compilation Error when HPX_MPI_WITH_FUTURES=ON
* :hpx-issue:`4537` - Linking issue with libhpx_initd.a
* :hpx-issue:`4535` - API for checking if pool with a given name exists
* :hpx-issue:`4523` - Build of PR #4311 (git tag 9955e8e) fails
* :hpx-issue:`4519` - Documentation problem
* :hpx-issue:`4513` - HPXConfig.cmake contains ill-formed paths when library paths use backslashes
* :hpx-issue:`4507` - User-polling introduced by MPI futures module should be more generally usable
* :hpx-issue:`4506` - Make sure force_linking.hpp is not included in main module header
* :hpx-issue:`4501` - Fix compilation of PAPI tests
* :hpx-issue:`4497` - Add modules CI checks
* :hpx-issue:`4489` - Polymorphic executor
* :hpx-issue:`4476` - Use CMake targets defined by FindBoost
* :hpx-issue:`4473` - Add vcpkg installation instructions
* :hpx-issue:`4470` - Adapt hpx::future to C++20 co_await
* :hpx-issue:`4468` - Compile error on Raspberry Pi 4
* :hpx-issue:`4466` - Compile error on Windows, current stable:
* :hpx-issue:`4453` - Installing HPX on fedora with dnf is not adding cmake files
* :hpx-issue:`4448` - New std::variant serialization broken
* :hpx-issue:`4438` - Add performance counter flag is monotically increasing
* :hpx-issue:`4436` - Build problem: same code build and works with 1.4.0 but it doesn't with 1.4.1
* :hpx-issue:`4429` - Function descriptions not supported in distributed
* :hpx-issue:`4423` - --hpx:ini=hpx.lock_detection=0 has no effect
* :hpx-issue:`4422` - Add performance counter metadata
* :hpx-issue:`4419` - Weird behavior for --hpx:print-counter-interval with large numbers
* :hpx-issue:`4401` - Create module repository
* :hpx-issue:`4400` - Command line options conflict related to performance counters
* :hpx-issue:`4349` - `--hpx:use-process-mask` option throw an exception on OS X
* :hpx-issue:`4345` - Move gh-pages branch out of hpx repo
* :hpx-issue:`4323` - Const-correctness error in assignment operator of compute::vector
* :hpx-issue:`4318` - ASIO breaks with C++2a concepts
* :hpx-issue:`4317` - Application runs even if `--hpx:help` is specified
* :hpx-issue:`4063` - Document hpxcxx compiler wrapper
* :hpx-issue:`3983` - Implement the C++20 Synchronization Library
* :hpx-issue:`3696` - C++11 `constexpr` support is now required
* :hpx-issue:`3623` - Modular HPX branch and an alternative project layout
* :hpx-issue:`2836` - The worst-case time complexity of parallel::sort seems to be O(N^2).

Closed pull requests
====================

* :hpx-pr:`4936` - Minor documentation fixes part 2
* :hpx-pr:`4935` - Add copyright and license to joss paper file
* :hpx-pr:`4934` - Adding Semicolon in Documentation
* :hpx-pr:`4932` - Fixing compiler warnings
* :hpx-pr:`4931` - Small documentation formatting fixes
* :hpx-pr:`4930` - Documentation Distributed HPX applications localvv with local_vv
* :hpx-pr:`4929` - Add final version of the JOSS paper
* :hpx-pr:`4928` - Add HPX_NODISCARD to enable_user_polling structs
* :hpx-pr:`4926` - Rename distributed_executors module to executors_distributed
* :hpx-pr:`4925` - Making transform_reduce conforming to C++20
* :hpx-pr:`4923` - Don't acquire lock if not needed
* :hpx-pr:`4921` - Update the release notes for the release candidate 3
* :hpx-pr:`4920` - Disable libcds release
* :hpx-pr:`4919` - Make cuda event pool dynamic instead of fixed size
* :hpx-pr:`4917` - Move chrono functionality to hpx::chrono namespace
* :hpx-pr:`4916` - HPX_HAVE_DEPRECATION_WARNINGS needs to be set even when disabled
* :hpx-pr:`4915` - Moving more action related files to actions modules
* :hpx-pr:`4914` - Add alias targets with namespaces used for exporting
* :hpx-pr:`4912` - Aggregate initialize CPOs
* :hpx-pr:`4910` - Explicitly specify hwloc root on Jenkins CSCS builds
* :hpx-pr:`4908` - Fix algorithms documentation
* :hpx-pr:`4907` - Remove HPX::hpx_no_wrap_main target
* :hpx-pr:`4906` - Fixing unused variable warning
* :hpx-pr:`4905` - Adding specializations for simple for_loops
* :hpx-pr:`4904` - Update boost to 1.74.0 for the newest jenkins configs
* :hpx-pr:`4903` - Hide GITHUB_TOKEN environment variables from environment variable output
* :hpx-pr:`4902` - Cancel previous pull requests builds before starting a new one with Jenkins
* :hpx-pr:`4901` - Update public API list with updated algorithms
* :hpx-pr:`4899` - Suggested changes for HPX V1.5 release notes
* :hpx-pr:`4898` - Minor tweak to hpx::equal implementation
* :hpx-pr:`4896` - Making generate() and generate_n conforming to C++20
* :hpx-pr:`4895` - Update apex tag
* :hpx-pr:`4894` - Fix exception handling for tasks
* :hpx-pr:`4893` - Remove last use of std::result_of, removed in C++20
* :hpx-pr:`4892` - Adding replay_executor and replicate_executor
* :hpx-pr:`4889` - Restore old behaviour of not requiring linking to hpx_wrap when HPX_WITH_DYNAMIC_HPX_MAIN=OFF
* :hpx-pr:`4887` - Making sure remotely thrown (non-hpx) exceptions are properly marshaled back to invocation site
* :hpx-pr:`4885` - Adapting hpx::find and friends to C++20
* :hpx-pr:`4884` - Adapting mismatch to C++20
* :hpx-pr:`4883` - Adapting hpx::equal to be conforming to C++20
* :hpx-pr:`4882` - Fixing exception handling for hpx::copy and adding missing tests
* :hpx-pr:`4881` - Adds different runtime exception when registering thread with the HPX runtime
* :hpx-pr:`4876` - Adding example demonstrating how to disable thread stealing during the execution of parallel algorithms
* :hpx-pr:`4874` - Adding non-policy tests to all_of, any_of, and none_of
* :hpx-pr:`4873` - Set CUDA compute capability on rostam Jenkins builds
* :hpx-pr:`4872` - Force partitioned vector scan tests to run serially
* :hpx-pr:`4870` - Making move conforming with C++20
* :hpx-pr:`4869` - Making destroy and destroy_n conforming to C++20
* :hpx-pr:`4868` - Fix miscellaneous header problems
* :hpx-pr:`4867` - Add CPOs for for_each
* :hpx-pr:`4865` - Adapting count and count_if to be conforming to C++20
* :hpx-pr:`4864` - Release notes 1.5.0
* :hpx-pr:`4863` - adding libcds-hpx tag to prepare for hpx1.5 release
* :hpx-pr:`4862` - Adding version specific deprecation options
* :hpx-pr:`4861` - Limiting executor improvements
* :hpx-pr:`4860` - Making fill and fill_n compatible with C++20
* :hpx-pr:`4859` - Adapting all_of, any_of, and none_of to C++20
* :hpx-pr:`4857` - Improve libCDS integration
* :hpx-pr:`4856` - Correct typos in the documentation of the hpx performance counters
* :hpx-pr:`4854` - Removing obsolete code
* :hpx-pr:`4853` - Adding test that derives component from two other components
* :hpx-pr:`4852` - Fix mpi_ring test in distributed mode by ensuring all ranks run hpx_main
* :hpx-pr:`4851` - Converting resiliency APIs to tag_invoke based CPOs
* :hpx-pr:`4849` - Enable use of future_overhead test when DISTRIBUTED_RUNTIME is OFF
* :hpx-pr:`4847` - Fixing 'error prone' constructs as reported by Codacy
* :hpx-pr:`4846` - Disable Boost.Asio concepts support
* :hpx-pr:`4845` - Fix PAPI counters
* :hpx-pr:`4843` - Remove dependency on various Boost headers
* :hpx-pr:`4841` - Rearrange public API headers
* :hpx-pr:`4840` - Fixing TSS problems during thread termination
* :hpx-pr:`4839` - Fix async_cuda build problems when distributed runtime is disabled
* :hpx-pr:`4837` - Restore compatibility for old (now deprecated) copy algorithms
* :hpx-pr:`4836` - Adding CPOs for hpx::reduce
* :hpx-pr:`4835` - Remove `using util::result_of` from namespace hpx
* :hpx-pr:`4834` - Fixing the calculation of the number of idle cores and the corresponding idle masks
* :hpx-pr:`4833` - Allow thread function destructors to yield
* :hpx-pr:`4832` - Fixing assertion in split_gids and memory leaks in 1d_stencil_7
* :hpx-pr:`4831` - Making sure MPI_CXX_COMPILE_FLAGS is interpreted as a sequence  of options
* :hpx-pr:`4830` - Update documentation on using HPX::wrap_main
* :hpx-pr:`4827` - Update clang-newest configuration to use clang 10
* :hpx-pr:`4826` - Add Jenkins configuration for rostam
* :hpx-pr:`4825` - Move all CUDA functionality to hpx::cuda::experimental namespace
* :hpx-pr:`4824` - Add support for building master/release branches to Jenkins configuration
* :hpx-pr:`4821` - Implement customization point for hpx::copy and hpx::ranges::copy
* :hpx-pr:`4819` - Allow finding Boost components before finding HPX
* :hpx-pr:`4817` - Adding range version of stable sort
* :hpx-pr:`4815` - Fix a wrong #ifdef for IO/TIMER pools causing build errors
* :hpx-pr:`4814` - Replace hpx::function_nonser with std::function in error module
* :hpx-pr:`4809` - Foreach adapt
* :hpx-pr:`4808` - Make internal algorithms functions const
* :hpx-pr:`4807` - Add Jenkins configuration for running on Piz Daint
* :hpx-pr:`4806` - Update documentation links to new domain name
* :hpx-pr:`4805` - Applying changes that resolve time complexity issues in sort
* :hpx-pr:`4803` - Adding implementation of stable_sort
* :hpx-pr:`4802` - Fix datapar header paths
* :hpx-pr:`4801` - Replace boost::shared_array<T> with std::shared_ptr<T[]> if supported
* :hpx-pr:`4799` - Fixing #include paths in compatibility headers
* :hpx-pr:`4798` - Include the main module header (fixes partially #4488)
* :hpx-pr:`4797` - Change cmake targets
* :hpx-pr:`4794` - Removing 128bit integer emulation
* :hpx-pr:`4793` - Make sure global variable is handled properly
* :hpx-pr:`4792` - Replace enable_if with HPX_CONCEPT_REQUIRES_ and add is_sentinel_for constraint
* :hpx-pr:`4790` - Move deprecation warnings from base template to template specializations for result_of etc. structs
* :hpx-pr:`4789` - Fix hangs during assertion handling and distributed runtime construction
* :hpx-pr:`4788` - Fixing inclusive transform scan algorithm to properly handle initial value
* :hpx-pr:`4785` - Fixing barrier test
* :hpx-pr:`4784` - Fixing deleter argument bindings in serialize_buffer
* :hpx-pr:`4783` - Add coveralls badge
* :hpx-pr:`4782` - Make header tests parallel again
* :hpx-pr:`4780` - Remove outdated comment about hpx::stop in documentation
* :hpx-pr:`4776` - debug print improvements
* :hpx-pr:`4775` - Checkpoint cleanup
* :hpx-pr:`4771` - Fix compilation with HPX_WITH_NETWORKING=OFF
* :hpx-pr:`4767` - Remove all force linking leftovers
* :hpx-pr:`4765` - Fix 1d stencil index calculation
* :hpx-pr:`4764` - Force some tests to run serially
* :hpx-pr:`4762` - Update pointees in compatibility headers
* :hpx-pr:`4761` - Fix running and building of execution module tests on CircleCI
* :hpx-pr:`4760` - Storing hpx_options in global property to speed up summary report
* :hpx-pr:`4759` - Reduce memory requirements for our main shared state
* :hpx-pr:`4757` - Fix mimalloc linking on Windows
* :hpx-pr:`4756` - Fix compilation issues
* :hpx-pr:`4753` - Re-adding API functions that were lost during merges
* :hpx-pr:`4751` - Revert "Create coverage reports and upload them to codecov.io"
* :hpx-pr:`4750` - Fixing possible race condition during termination detection
* :hpx-pr:`4749` - Deprecate result_of and friends
* :hpx-pr:`4748` - Create coverage reports and upload them to codecov.io
* :hpx-pr:`4747` - Changing #include for MPI parcelport
* :hpx-pr:`4745` - Add `is_sentinel_for` trait implementation and test
* :hpx-pr:`4743` - Fix init_globally example after runtime mode changes
* :hpx-pr:`4742` - Update SUPPORT.md
* :hpx-pr:`4741` - Fixing a warning generated for unity builds with msvc
* :hpx-pr:`4740` - Rename local_lcos and basic_execution modules
* :hpx-pr:`4739` - Undeprecate a couple of hpx/modulename.hpp headers
* :hpx-pr:`4738` - Conditionally test schedulers in thread_stacksize_current test
* :hpx-pr:`4734` - Fixing a bunch of codacy warnings
* :hpx-pr:`4733` - Add experimental unity build option to CMake configuration
* :hpx-pr:`4730` - Fixing compilation problems with unordered map
* :hpx-pr:`4729` - Fix APEX build
* :hpx-pr:`4727` - Fix missing runtime includes for distributed runtime
* :hpx-pr:`4726` - Add more API headers
* :hpx-pr:`4725` - Add more compatibility headers for deprecated module headers
* :hpx-pr:`4724` - Fix 4723
* :hpx-pr:`4721` - Attempt to fixing migration tests
* :hpx-pr:`4717` - Make the compatilibility headers macro conditional
* :hpx-pr:`4716` - Add hpx/runtime.hpp and hpx/distributed/runtime.hpp API headers
* :hpx-pr:`4714` - Add hpx/future.hpp header
* :hpx-pr:`4713` - Remove hpx/runtime/threads_fwd.hpp and hpx/util_fwd.hpp
* :hpx-pr:`4711` - Make module deprecation warnings overridable
* :hpx-pr:`4710` - Add compatibility headers and other fixes after module header renaming
* :hpx-pr:`4708` - Add termination handler for parallel algorithms
* :hpx-pr:`4707` - Use hpx::function_nonser instead of std::function internally
* :hpx-pr:`4706` - Move header file to module
* :hpx-pr:`4705` - Fix incorrect behaviour of cmake-format check
* :hpx-pr:`4704` - Fix resource tests
* :hpx-pr:`4701` - Fix missing includes for future::then specializations
* :hpx-pr:`4700` - Removing obsolete memory component
* :hpx-pr:`4699` - Add short descriptions to modules missing documentation
* :hpx-pr:`4696` - Rename generated modules headers
* :hpx-pr:`4693` - Overhauling thread_mapper for public consumption
* :hpx-pr:`4688` - Fix thread stack size handling
* :hpx-pr:`4687` - Adding all_gather and fixing all_to_all
* :hpx-pr:`4684` - Miscellaneous compilation fixes
* :hpx-pr:`4683` - Fix HPX_WITH_DYNAMIC_HPX_MAIN=OFF
* :hpx-pr:`4682` - Fix compilation of pack_traversal_rebind_container.hpp
* :hpx-pr:`4681` - Add missing hpx/execution.hpp includes for future::then
* :hpx-pr:`4678` - Typeless communicator
* :hpx-pr:`4677` - Forcing registry option to be accepted without checks.
* :hpx-pr:`4676` - Adding scatter_to/scatter_from collective operations
* :hpx-pr:`4673` - Fix PAPI counters compilation
* :hpx-pr:`4671` - Deprecate hpx::promise alias to hpx::lcos::promise
* :hpx-pr:`4670` - Explicitly instantiate get_exception
* :hpx-pr:`4667` - Add `stopValue` in `Sentinel` struct instead of `Iterator`
* :hpx-pr:`4666` - Add release build on Windows to GitHub actions
* :hpx-pr:`4664` - Creating itt_notify module.
* :hpx-pr:`4663` - Mpi fixes
* :hpx-pr:`4659` - Making sure declarations match definitions in register_locks implementation
* :hpx-pr:`4655` - Fixing task annotations for actions
* :hpx-pr:`4653` - Making sure APEX is linked into every application, if needed
* :hpx-pr:`4651` - Update get_function_annotation.hpp
* :hpx-pr:`4646` - Runtime type
* :hpx-pr:`4645` - Add a few more API headers
* :hpx-pr:`4644` - Fixing support for mpirun (and similar)
* :hpx-pr:`4643` - Fixing the fix for get_idle_core_count() API
* :hpx-pr:`4638` - Remove HPX_API_EXPORT missed in previous cleanup
* :hpx-pr:`4636` - Adding C++20 barrier
* :hpx-pr:`4635` - Adding C++20 latch API
* :hpx-pr:`4634` - Adding C++20 counting semaphore API
* :hpx-pr:`4633` - Unify execution parameters customization points
* :hpx-pr:`4632` - Adding missing bulk_sync_execute wrapper to example executor
* :hpx-pr:`4631` - Updates to documentation; grammar edits.
* :hpx-pr:`4630` - Updates to documentation; moved hyperlink
* :hpx-pr:`4624` - Export set_self_ptr in thread_data.hpp instead of with forward declarations where used
* :hpx-pr:`4623` - Clean up export macros
* :hpx-pr:`4621` - Trigger an error for older boost versions on power architectures
* :hpx-pr:`4617` - Ignore user-set compatibility header options if the module does not have compatibility headers
* :hpx-pr:`4616` - Fix cmake-format warning
* :hpx-pr:`4615` - Add handler for serializing custom exceptions
* :hpx-pr:`4614` - Fix error message when HPX_IGNORE_CMAKE_BUILD_TYPE_COMPATIBILITY=OFF
* :hpx-pr:`4613` - Make partitioner constructor private
* :hpx-pr:`4611` - Making auto_chunk_size execute the given function using the given executor
* :hpx-pr:`4610` - Making sure the thread-local lock registration data is moving to the core the suspended HPX thread is resumed on
* :hpx-pr:`4609` - Adding an API function that exposes the number of idle cores
* :hpx-pr:`4608` - Fixing moodycamel namespace
* :hpx-pr:`4607` - Moving winsocket initialization to core library
* :hpx-pr:`4606` - Local runtime module etc.
* :hpx-pr:`4604` - Add config_registry module
* :hpx-pr:`4603` - Deal with distributed modules in their respective CMakeLists.txt
* :hpx-pr:`4602` - Small module fixes
* :hpx-pr:`4598` - Making sure current_executor and service_executor functions are linked into the core library
* :hpx-pr:`4597` - Adding broadcast_to/broadcast_from to collectives module
* :hpx-pr:`4596` - Fix performance regression in block_executor
* :hpx-pr:`4595` - Making sure main.cpp is built as a library if HPX_WITH_DYNAMIC_MAIN=OFF
* :hpx-pr:`4592` - Futures module
* :hpx-pr:`4591` - Adapting co_await support for C++20
* :hpx-pr:`4590` - Adding missing exception test for for_loop()
* :hpx-pr:`4587` - Move traits headers to hpx/modulename/traits directory
* :hpx-pr:`4586` - Remove Travis CI config
* :hpx-pr:`4585` - Update macOS test blacklist
* :hpx-pr:`4584` - Attempting to fix missing symbols in stack trace
* :hpx-pr:`4583` - Fixing bad static_cast
* :hpx-pr:`4582` - Changing download url for Windows prerequisites to circumvent bandwidth limitations
* :hpx-pr:`4581` - Adding missing using placeholder::_X
* :hpx-pr:`4579` - Move get_stack_size_name and related functions
* :hpx-pr:`4575` - Excluding unconditional definition of class backtrace from global header
* :hpx-pr:`4574` - Changing return type of hardware_concurrency() to unsigned int
* :hpx-pr:`4570` - Move tests to modules
* :hpx-pr:`4564` - Reshuffle internal targets and add HPX::hpx_no_wrap_main target
* :hpx-pr:`4563` - fix CMake option typo
* :hpx-pr:`4562` - Unregister lock earlier to avoid holding it while suspending
* :hpx-pr:`4561` - Adding test macros supporting custom output stream
* :hpx-pr:`4560` - Making sure hash_any::operator()() is linked into core library
* :hpx-pr:`4559` - Fixing compilation if HPX_WITH_THREAD_BACKTRACE_ON_SUSPENSION=On
* :hpx-pr:`4557` - Improve spinlock implementation to perform better in high-contention situations
* :hpx-pr:`4553` - Fix a runtime_ptr problem at shutdown when apex is enabled
* :hpx-pr:`4552` - Add configuration option for making exceptions less noisy
* :hpx-pr:`4551` - Clean up thread creation parameters
* :hpx-pr:`4549` - Test FetchContent build on GitHub actions
* :hpx-pr:`4548` - Fix stack size
* :hpx-pr:`4545` - Fix header tests
* :hpx-pr:`4544` - Fix a typo in sanitizer build
* :hpx-pr:`4541` - Add API to check if a thread pool exists
* :hpx-pr:`4540` - Making sure MPI support is enabled if MPI futures are used but networking is disabled
* :hpx-pr:`4538` - Move channel documentation examples to examples directory
* :hpx-pr:`4536` - Add generic allocator for execution policies
* :hpx-pr:`4534` - Enable compatibility headers for thread_executors module
* :hpx-pr:`4532` - Fixing broken url in README.rst
* :hpx-pr:`4531` - Update scripts
* :hpx-pr:`4530` - Make sure module API docs show up in correct order
* :hpx-pr:`4529` - Adding missing template code to module creation script
* :hpx-pr:`4528` - Make sure version module uses HPX's binary dir, not the parent's
* :hpx-pr:`4527` - Creating actions_base and actions module
* :hpx-pr:`4526` - Shared state for cv
* :hpx-pr:`4525` - Changing sub-name sequencing for experimental namespace
* :hpx-pr:`4524` - Add API guarantee notes to API reference documentation
* :hpx-pr:`4522` - Enable and fix deprecation warnings in execution module
* :hpx-pr:`4521` - Moves more miscellaneous files to modules
* :hpx-pr:`4520` - Skip execution customization points when executor is known
* :hpx-pr:`4518` - Module distributed lcos
* :hpx-pr:`4516` - Fix various builds
* :hpx-pr:`4515` - Replace backslashes by slashes in windows paths
* :hpx-pr:`4514` - Adding polymorphic_executor
* :hpx-pr:`4512` - Adding C++20 jthread and stop_token
* :hpx-pr:`4510` - Attempt to fix APEX linking in external packages again
* :hpx-pr:`4508` - Only test pull requests (not all branches) with GitHub actions
* :hpx-pr:`4505` - Fix duplicate linking in tests (ODR violations)
* :hpx-pr:`4504` - Fix C++ standard handling
* :hpx-pr:`4503` - Add CMakelists file check
* :hpx-pr:`4500` - Fix .clang-format version requirement comment
* :hpx-pr:`4499` - Attempting to fix hpx_init linking on macOS
* :hpx-pr:`4498` - Fix compatibility of `pool_executor`
* :hpx-pr:`4496` - Removing superfluous SPDX tags
* :hpx-pr:`4494` - Module executors
* :hpx-pr:`4493` - Pack traversal module
* :hpx-pr:`4492` - Update copyright year in documentation
* :hpx-pr:`4491` - Add missing current_executor header
* :hpx-pr:`4490` - Update GitHub actions configs
* :hpx-pr:`4487` - Properly dispatch exceptions thrown from hpx_main to be rethrown from hpx::init/hpx::stop
* :hpx-pr:`4486` - Fixing an initialization order problem
* :hpx-pr:`4485` - Move miscellaneous files to their rightful modules
* :hpx-pr:`4483` - Clean up imported CMake target naming
* :hpx-pr:`4481` - Add vcpkg installation instructions
* :hpx-pr:`4479` - Add hints to allow to specify MIMALLOC_ROOT
* :hpx-pr:`4478` - Async modules
* :hpx-pr:`4475` - Fix rp init changes
* :hpx-pr:`4474` - Use #pragma once in headers
* :hpx-pr:`4472` - Add more descriptive error message when using x86 coroutines on non-x86 platforms
* :hpx-pr:`4467` - Add mimalloc find cmake script
* :hpx-pr:`4465` - Add thread_executors module
* :hpx-pr:`4464` - Include module
* :hpx-pr:`4462` - Merge hpx_init and hpx_wrap into one static library
* :hpx-pr:`4461` - Making thread_data test more realistic
* :hpx-pr:`4460` - Suppress MPI warnings in version.cpp
* :hpx-pr:`4459` - Make sure pkgconfig applications link with hpx_init
* :hpx-pr:`4458` - Added example demonstrating how to create and use a wrapping executor
* :hpx-pr:`4457` - Fixing execution of thread exit functions
* :hpx-pr:`4456` - Move backtrace files to debugging module
* :hpx-pr:`4455` - Move deadlock_detection and maintain_queue_wait_times source files into schedulers module
* :hpx-pr:`4450` - Fixing compilation with std::filesystem enabled
* :hpx-pr:`4449` - Fixing build system to actually build variant test
* :hpx-pr:`4447` - This fixes an obsolete #include
* :hpx-pr:`4446` - Resume tasks where they were suspended
* :hpx-pr:`4444` - Minor CUDA fixes
* :hpx-pr:`4443` - Add missing tests to CircleCI config
* :hpx-pr:`4442` - Adding a tag to all auto-generated files allowing for tools to visually distinguish those
* :hpx-pr:`4441` - Adding performance counter type information
* :hpx-pr:`4440` - Fixing MSVC build
* :hpx-pr:`4439` - Link HPX::plugin and component privately in hpx_setup_target
* :hpx-pr:`4437` - Adding a test that verifies the problem can be solved using a trait specialization
* :hpx-pr:`4434` - Clean up Boost dependencies and copy string algorithms to new module
* :hpx-pr:`4433` - Fixing compilation issues (!) if MPI parcelport is enabled
* :hpx-pr:`4431` - Ignore warnings about name mangling changing
* :hpx-pr:`4430` - Add performance_counters module
* :hpx-pr:`4428` - Don't add compatibility headers to module API reference
* :hpx-pr:`4426` - Add currently failing tests on GitHub actions to blacklist
* :hpx-pr:`4425` - Clean up and correct minimum required versions
* :hpx-pr:`4424` - Making sure hpx.lock_detection=0 works as advertized
* :hpx-pr:`4421` - Making sure interval time stops underlying timer thread on termination
* :hpx-pr:`4417` - Adding serialization support for std::variant (if available) and std::tuple
* :hpx-pr:`4415` - Partially reverting changes applied by PR 4373
* :hpx-pr:`4414` - Added documentation for the compiler-wrapper script hpxcxx.in in creating_hpx_projects.rst
* :hpx-pr:`4413` - Merging from V1.4.1 release
* :hpx-pr:`4412` - Making sure to issue a warning if a file specified using --hpx:options-file is not found
* :hpx-pr:`4411` - Make test specific to HPX_WITH_SHARED_PRIORITY_SCHEDULER
* :hpx-pr:`4407` - Adding minimal MPI executor
* :hpx-pr:`4405` - Fix cross pool injection test, use default scheduler as falback
* :hpx-pr:`4404` - Fix a race condition and clean-up usage of scheduler mode
* :hpx-pr:`4399` - Add more threading modules
* :hpx-pr:`4398` - Add CODEOWNERS file
* :hpx-pr:`4395` - Adding a parameter to auto_chunk_size allowing to control the amount of iterations to measure
* :hpx-pr:`4393` - Use appropriate cache-line size defaults for different platforms
* :hpx-pr:`4391` - Fixing use of allocator for C++20
* :hpx-pr:`4390` - Making --hpx:help behavior consistent
* :hpx-pr:`4388` - Change the resource partitioner initialization
* :hpx-pr:`4387` - Fix roll_release.sh
* :hpx-pr:`4386` - Add warning messages for using thread binding options on macOS
* :hpx-pr:`4385` - Cuda futures
* :hpx-pr:`4384` - Make enabling dynamic hpx_main on non-Linux systems a configuration error
* :hpx-pr:`4383` - Use configure_file for HPXCacheVariables.cmake
* :hpx-pr:`4382` - Update spellchecking whitelist and fix more typos
* :hpx-pr:`4380` - Add a helper function to get a future from a cuda stream
* :hpx-pr:`4379` - Add Windows and macOS CI with GitHub actions
* :hpx-pr:`4378` - Change C++ standard handling
* :hpx-pr:`4377` - Remove Python scripts
* :hpx-pr:`4374` - Adding overload for `hpx::init`/`hpx::start` for use with resource partitioner
* :hpx-pr:`4373` - Adding test that verifies for 4369 to be fixed
* :hpx-pr:`4372` - Another attempt at fixing the integral mismatch and conversion warnings
* :hpx-pr:`4370` - Doc updates quick start
* :hpx-pr:`4368` - Add a whitelist of words for weird spelling suggestions
* :hpx-pr:`4366` - Suppress or fix clang-tidy-9 warnings
* :hpx-pr:`4365` - Removing more Boost dependencies
* :hpx-pr:`4363` - Update clang-format config file for version 9
* :hpx-pr:`4362` - Fix indices typo
* :hpx-pr:`4361` - Boost cleanup
* :hpx-pr:`4360` - Move plugins
* :hpx-pr:`4358` - Doc updates; generating documentation. Will likely need heavy editing.
* :hpx-pr:`4356` - Remove some minor unused and unnecessary Boost includes
* :hpx-pr:`4355` - Fix spellcheck step in CircleCI config
* :hpx-pr:`4354` - Lightweight utility to hold a pack as members
* :hpx-pr:`4352` - Minor fixes to the C++ standard detection for MSVC
* :hpx-pr:`4351` - Move generated documentation to hpx-docs repo
* :hpx-pr:`4347` - Add cmake policy - CMP0074
* :hpx-pr:`4346` - Remove file committed by mistake
* :hpx-pr:`4342` - Remove HCC and SYCL options from CMakeLists.txt
* :hpx-pr:`4341` - Fix launch process test with APEX enabled
* :hpx-pr:`4340` - Testing Cirrus CI
* :hpx-pr:`4339` - Post 1.4.0 updates
* :hpx-pr:`4338` - Spelling corrections and CircleCI spell check
* :hpx-pr:`4333` - Flatten bound callables
* :hpx-pr:`4332` - This is a collection of mostly minor (cleanup) fixes
* :hpx-pr:`4331` - This adds the missing tests for async_colocated and async_continue_colocated
* :hpx-pr:`4330` - Remove HPX.Compute host default_executor
* :hpx-pr:`4328` - Generate global header for basic_execution module
* :hpx-pr:`4327` - Use INTERNAL_FLAGS option for all examples and components
* :hpx-pr:`4326` - Usage of temporary allocator in assignment operator of compute::vector
* :hpx-pr:`4325` - Use hpx::threads::get_cache_line_size in prefetching.hpp
* :hpx-pr:`4324` - Enable compatibility headers option for execution module
* :hpx-pr:`4316` - Add clang format indentppdirectives
* :hpx-pr:`4313` - Introduce index_pack alias to pack of size_t
* :hpx-pr:`4312` - Fixing compatibility header for pack.hpp
* :hpx-pr:`4311` - Dataflow annotations for APEX
* :hpx-pr:`4309` - Update launching_and_configuring_hpx_applications.rst
* :hpx-pr:`4306` - Fix schedule hint not being taken from executor
* :hpx-pr:`4305` - Implementing `hpx::functional::tag_invoke`
* :hpx-pr:`4304` - Improve pack support utilities
* :hpx-pr:`4303` -  Remove errors module dependency on datastructures
* :hpx-pr:`4301` - Clean up thread executors
* :hpx-pr:`4294` - Logging revamp
* :hpx-pr:`4292` - Remove SPDX tag from Boost License file to allow for github to recognize it
* :hpx-pr:`4291` - Add format support for std::tm
* :hpx-pr:`4290` - Simplify compatible tuples check
* :hpx-pr:`4288` - A lightweight take on boost::lexical_cast
* :hpx-pr:`4287` - Forking boost::lexical_cast as a new module
* :hpx-pr:`4277` - MPI_futures
* :hpx-pr:`4270` - Refactor future implementation
* :hpx-pr:`4265` - Threading module
* :hpx-pr:`4259` - Module naming base
* :hpx-pr:`4251` - Local workrequesting scheduler
* :hpx-pr:`4250` - Inline execution of scoped tasks, if possible
* :hpx-pr:`4247` - Add execution in module headers
* :hpx-pr:`4246` - Expose CMake targets officially
* :hpx-pr:`4239` - Doc updates miscellaneous (partially completed during Google Season of Docs)
* :hpx-pr:`4233` - Remove project() from modules + fix CMAKE_SOURCE_DIR issue
* :hpx-pr:`4231` - Module local lcos
* :hpx-pr:`4207` - Command line handling module
* :hpx-pr:`4206` - Runtime configuration module
* :hpx-pr:`4141` - Doc updates examples local to remote (partially completed during Google Season of Docs)
* :hpx-pr:`4091` - Split runtime into local and distributed parts
* :hpx-pr:`4017` - Require C++14
