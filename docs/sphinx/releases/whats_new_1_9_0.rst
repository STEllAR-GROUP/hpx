..
    Copyright (C) 2007-2022 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_9_0:

===========================
|hpx| V1.9.0 (May 2, 2023)
===========================

General changes
===============
- Added RISC-V 64bit support. HPX is now compatible with RISC-V
  architectures which have revolutionized the HPC world.
- LCI parcelport has been optimized to transfer parcels with 
  fewer messages and use the HPX resource partitioner for 
  its progress thread allocation. It should generally provide 
  better performance than before. It also removes its dependency
  on the MPI library.
- HPX dependency on Boost was further relaxed by replacing headers
  from Boost.Range, Boost.Tokenizer and Boost.Lockfree.
- Improvements took place on our parallel algorithms implementation.
- Our Senders/Receivers (P2300) integration was extended:

  - Coroutines were integrated with senders/receivers.
  ``get_completion_signatures`` now works with awaitable senders.
  - ``with_awaitable_senders`` allows the passed senders to
  retrieve the value i.e. senders are transparently
  awaitable from within a coroutine.
  - ``when_all_vector`` was added.

- ``sync_wait`` and ``sync_wait_with_variant`` sender consumers were
  added. The user can now initiate the execution of
  their asynchronous pipeline by blocking the current thread that
  executes the main() function until the result is retrieved.
- The combinators for futures (a.k.a. async_combinators) ``when_*``,
  ``wait_*``, ``wait_*_nothrow`` were turned into CPOs allowing for 
  end-user customization. For more information on the async_combinators
  refer to the documentation,
  https://hpx-docs.stellar-group.org/latest/html/libs/core/async_combinators/docs/index.html?highlight=combinators.
- The new datapar backend SVE allows simd and par_simd execution policies
  to exploit dataparalleism in the processors that have SVE vector
  registers like A64FX and Neoverse V1.
- The documentation for parallel algorithms, container algorithms was
  further improved. The Public API page was vastly enriched. 
- Copy button shortkey was added at the top-right of code-blocks.
- Pragma directive that reports warnings as errors on MSVC was fixed. 
- Command line argument ``--hpx:loopback_network`` was added to
  facilitate debugging with networks.
- We added an HPX-SYCL integration, allowing users to obtain HPX futures
  for SYCL events. This effectively enables the integration of arbitrary
  asynchronous SYCL operations into the HPX task graph. Bolted on top 
  of this integration, we further added an HPX-SYCL executor for ease of use.

Breaking changes
================

- Stopped supporting Clang V8, the minimal version supported is now Clang V10.
- Stopped supporting gcc V8, the minimal version supported is now gcc V9.
- Stopped supporting Visual Studio 2015, the minimal version supported is
  now Visual Studio 2019.
- ``tag_policy_tag`` et.al. were re-added after HPX V1.8.1 depracation.
- ``get_chunk_size`` and ``processing_units_count`` API is now expecting
  the time for one iteration as an argument.
- The list of all the namespace changes can be found here: :ref:`new_namespaces_1_9_0`.

Closed issues
=============

* :hpx-issue:`6203` - Compilation error with `-mcpu=a64fx` on Ookami
* :hpx-issue:`6196` - Incorrect log destination
* :hpx-issue:`6191` - installing HPX 
* :hpx-issue:`6184` - Wrong processing_units_count of restricted_thread_pool_executor
* :hpx-issue:`6171` - Release Tag Name Request
* :hpx-issue:`6162` - Current master does not compile on ROSTAM
* :hpx-issue:`6156` - hpxcxx does not work if HPX_WITH_PKGCONFIG=OFF
* :hpx-issue:`6108` - cxx17_aligned_new.cpp on msvc fails due to wrong pragma directive
* :hpx-issue:`6045` - Can't call nullary callables wrapped with `hpx::unwrapping`
* :hpx-issue:`6013` - Unable to build subprojects hpx_collectives/hpx_compute with MSVC
* :hpx-issue:`6008` - Missing `constexpr` default constructor for `hpx::mutex`
* :hpx-issue:`5999` - Add HPX Conda package to conda-forge
* :hpx-issue:`5998` - Serializing multiple arguments when applying distributed action results in segfault
* :hpx-issue:`5958` - HPX 1.8.0 and Blaze issues
* :hpx-issue:`5908` - Windows: duplicated symbols in static builds
* :hpx-issue:`5802` - Lost status is_ready from future
* :hpx-issue:`5767` - Performance drop on Piz Daint
* :hpx-issue:`5752` - Implement stride_view from P1899 (experimental)
* :hpx-issue:`5744` - HPX_WITH_FETCH_ASIO not working on Ookami
* :hpx-issue:`5561` - Possible race condition in helper thread / hpx::cout

Closed pull requests
====================

* :hpx-pr:`6228` - Fixing algorithms for zero length sequences when run with s/r scheduler
* :hpx-pr:`6227` - Reliably disable background work when no networking is enabled
* :hpx-pr:`6225` - Make heap fails in par for small sized heaps #6217
* :hpx-pr:`6222` - Add documentation for `hpx::post`
* :hpx-pr:`6221` - Fix segmented algorithms tests
* :hpx-pr:`6218` - Creating INSTALL component 'runtime' to enable installing binaries only
* :hpx-pr:`6216` - added tests for set_difference, updated set_operation.hpp to fix #6198
* :hpx-pr:`6213` - Modernize and streamline MPI parcelport
* :hpx-pr:`6211` - Modernize modules of level 11, 12, and 13
* :hpx-pr:`6210` - Fixing MPI parcelport initialization if MPI is initialized outside of HPX
* :hpx-pr:`6209` - Prevent thread stealing during scheduler shutdown
* :hpx-pr:`6208` - Fix the compilation warning in the MPI parcelport with gcc 11.2
* :hpx-pr:`6207` - Automatically enable Boost.Context when compiling for arm64.
* :hpx-pr:`6206` - Update CMakeLists.txt
* :hpx-pr:`6205` - Do not generate hpxcxx if support for pkgconfig was disabled
* :hpx-pr:`6204` - Use LRT_ instead of LAPP_ logging in barrier implementation
* :hpx-pr:`6202` - Fixing Fedora build errors on Power systems
* :hpx-pr:`6201` - Update the LCI parcelport documents
* :hpx-pr:`6200` - Par link jobs
* :hpx-pr:`6197` - LCI parcelport: add doc, upgrade to v1.7.4, refactor cmake autofetch.
* :hpx-pr:`6195` - Change the default tag of autofetch LCI to v1.7.3.
* :hpx-pr:`6192` - Fix page `Writing single-node applications`
* :hpx-pr:`6189` - Making sure restricted_thread_pool_executor properly reports used number of cores
* :hpx-pr:`6187` - Enable using for_loop with range generators
* :hpx-pr:`6186` - thread_support/CMakeLists: Fix build issue
* :hpx-pr:`6185` - Fix EVE datapar with cxx_standard less than 20
* :hpx-pr:`6183` - Update CI integration for EVE
* :hpx-pr:`6182` - Fixing performance regressions
* :hpx-pr:`6181` - LCI parcelport: backlog queue, aggregation, separate devices, and more
* :hpx-pr:`6180` - Fixing use of for_loop with rebound execution policy (using `.with()`)
* :hpx-pr:`6179` - Taking predicates for algorithms by value
* :hpx-pr:`6178` - Changes needed to make chapel_hpx examples work
* :hpx-pr:`6176` - Fixing warnings that were generated by PVS Studio
* :hpx-pr:`6174` - Replace boost::integer::gcd with std::gcd
* :hpx-pr:`6172` - [Docs] Fix example of how to run single/specific test(s)
* :hpx-pr:`6170` - Adding missing fallback for processing_units_count customization point
* :hpx-pr:`6169` - LCI parcelport: bypass the parcel queue and connection cache.
* :hpx-pr:`6167` - Add create_local_communicator API function
* :hpx-pr:`6166` - Add missing header for std::intmax_t
* :hpx-pr:`6165` - Attempt to work around MSVC problem
* :hpx-pr:`6161` - Update EVE integration
* :hpx-pr:`6160` - More cleanup for module levels 0 to 10
* :hpx-pr:`6159` - Fix minor spelling mistake in generate_issue_pr_list.sh
* :hpx-pr:`6158` - Update documentation in `writing single-node applications` page 
* :hpx-pr:`6157` - Improve index_queue_spawning
* :hpx-pr:`6154` - Avoid performing late command line handling twice in distributed runtime
* :hpx-pr:`6152` - The -rd and -mr options didn't work, and they should have been --rd and --mr
* :hpx-pr:`6151` - Refactoring the Manual page in documentation
* :hpx-pr:`6148` - Investigate the failure of the LCI parcelport.
* :hpx-pr:`6147` - Make posix co-routine stacks non-executable
* :hpx-pr:`6146` - Avoid ambiguities wrt tag_invoke
* :hpx-pr:`6144` - General improvements to scheduling and related fixes
* :hpx-pr:`6143` - Add list of new namespaces for new release
* :hpx-pr:`6140` - Fixing background scheduler to properly exit in the end
* :hpx-pr:`6139` - [P2300] execution: Cleanup coroutines integration and improve ADL isolation
* :hpx-pr:`6137` - Adding example of a simple master/slave distributed application
* :hpx-pr:`6136` - Deprecate `execution::experimental::task_group` in favor of `experimental::task_group`
* :hpx-pr:`6135` - Fixing warnings reported by MSVC analysis
* :hpx-pr:`6134` - Adding notification function for parcelports to be called after early parcel handling
* :hpx-pr:`6132` - Fixing to_non_par() for parallel simd policies
* :hpx-pr:`6131` - modernize modules from level 25
* :hpx-pr:`6130` - Remove the mutex lock in the critical path of get_partitioner.
* :hpx-pr:`6129` - Modernize module from levels 22, 23
* :hpx-pr:`6127` - Working around gccV9 problem that prevent us from storing enum classes in bit fields
* :hpx-pr:`6126` - Deprecate hpx::parallel::task_block in favor of hpx::experimental::task_block
* :hpx-pr:`6125` - Making sure sync_wait compiles when used with an lvalue sender involving bulk
* :hpx-pr:`6124` - Fixing use of any_sender in combination with when_all
* :hpx-pr:`6123` - Fixed issues found by PVS-Studio
* :hpx-pr:`6121` - Modernize modules of level 21, 22
* :hpx-pr:`6120` - Use index_queue for parallel executors bulk_async_execute
* :hpx-pr:`6119` - Update CMakeLists.txt
* :hpx-pr:`6118` - Modernize modules from level 17, 18, 19, and 20
* :hpx-pr:`6117` - Initialize buffer_allocate_time_ to 0
* :hpx-pr:`6116` - Add new command line argument --hpx:loopback_network
* :hpx-pr:`6115` - Modernize modules of levels 14, 15, and 16
* :hpx-pr:`6114` - Enhance the formatting of the documentation 
* :hpx-pr:`6113` - Modernize modules in module level 11, 12, and 13
* :hpx-pr:`6112` - Modernize modules from levels 9 and 10
* :hpx-pr:`6111` - Modernize all modules from module level 8
* :hpx-pr:`6110` - Use pragma error directive to report warnings as errors on msvc
* :hpx-pr:`6109` - Modernize serialization module
* :hpx-pr:`6107` - Modernize error module
* :hpx-pr:`6106` - Modernizing modules of levels 0 to 5
* :hpx-pr:`6105` - Optimizations on LCI parcelport: merge small messages; remove sender mutex lock.
* :hpx-pr:`6104` - Adding parameters API: measure_iteration
* :hpx-pr:`6103` - Document `task_group` and include in Public API
* :hpx-pr:`6102` - Prevent warnings generated by clang-cl
* :hpx-pr:`6101` - Using more fold expressions
* :hpx-pr:`6100` - Deprecate `hpx::parallel::reduce_by_key` in favor of `hpx::experimental::reduce_by_key`
* :hpx-pr:`6098` - Forking Boost.Lockfree
* :hpx-pr:`6096` - Forking Boost.Tokenizer
* :hpx-pr:`6095` - Replacing facilities from Boost.Range
* :hpx-pr:`6094` - Removing object_semaphore
* :hpx-pr:`6093` - Replace boost::string_ref with std::string_view
* :hpx-pr:`6092` - Use C++17 static_assert where possible
* :hpx-pr:`6091` - Replace artificial sequencing with fold expressions
* :hpx-pr:`6090` - Fixing use of get_chunk_size customization point
* :hpx-pr:`6088` - Add/fix Public API documentation
* :hpx-pr:`6086` - Deprecate `hpx::util::unlock_guard` in favor of `hpx::unlock_guard`
* :hpx-pr:`6085` - Add experimental sycl integration/executor
* :hpx-pr:`6084` - Renaming hpx::apply and friends to hpx::post
* :hpx-pr:`6083` - Using if constexpr instead of tag-dispatching, where possible
* :hpx-pr:`6082` - Replace util::always_void_t with std::void_t
* :hpx-pr:`6081` - Update github actions to avoid warnings
* :hpx-pr:`6080` - Disable some tests that fail on LCI
* :hpx-pr:`6079` - Adding more natvis files, correct existing
* :hpx-pr:`6078` - Changing target name of memory_counters component
* :hpx-pr:`6077` - Making default constructor of hpx::mutex constexpr
* :hpx-pr:`6076` - Cleaning up functionality that was deprecated in V1.7
* :hpx-pr:`6075` - Remove conditional code for gcc V7 and below
* :hpx-pr:`6074` - Fixing compilation issues on gcc V8
* :hpx-pr:`6073` - Fixing PAPI counter component compilation
* :hpx-pr:`6072` - Adding ex::when_all_vector
* :hpx-pr:`6071` - Making get_forward_progress_guarantee_t specializations constexpr
* :hpx-pr:`6070` - Implement P2690 for our algorithms
* :hpx-pr:`6069` - Do not check for cancellation during each iteration but only once per partition
* :hpx-pr:`6068` - Prevent using task and non_task as a CPO
* :hpx-pr:`6067` - Deprecated hpx::util::mem_fn in favor of hpx::mem_fn
* :hpx-pr:`6066` - Create codeql.yml
* :hpx-pr:`6064` - Adapting adjacent_difference for S/R execution
* :hpx-pr:`6063` - Modernize iterator_support module
* :hpx-pr:`6062` - Make sure wrapping executor does not go out of scope prematurely
* :hpx-pr:`6061` - Minor fix in small_vector (from upstream)
* :hpx-pr:`6060` - Allow to disable registering signal handlers
* :hpx-pr:`6059` - [P2300] Fix: declval cannot be ODR used
* :hpx-pr:`6058` - Avoid ambiguity for hpx::get used with std::variant
* :hpx-pr:`6057` - Create a dedicated thread pool to run LCI_progress.
* :hpx-pr:`6056` - Fix coroutine test for clang
* :hpx-pr:`6055` - Patches needed to be able to build HPX 1.8.1 on various platforms
* :hpx-pr:`6054` - Use MSVC specific attribute [[msvc::no_unique_address]]
* :hpx-pr:`6052` - Deprecated hpx::util::invoke_fused in favor of hpx::invoke_fused
* :hpx-pr:`6051` - Add non-contiguous index queue and use it in thread_pool_bulk_scheduler
* :hpx-pr:`6049` - Crosscompile arm sve
* :hpx-pr:`6048` - Deprecated hpx::util::invoke in favor of hpx::invoke
* :hpx-pr:`6047` - Separating binary_semaphore into its own file
* :hpx-pr:`6046` - Support using unwrapping with nullary function objects
* :hpx-pr:`6044` - Generalize the use of then() and dataflow
* :hpx-pr:`6043` - Clean up scan_partitioner
* :hpx-pr:`6042` - Modernize dataflow API
* :hpx-pr:`6041` - docs: document semaphores
* :hpx-pr:`6040` - Add/Fix documentation of Public API page
* :hpx-pr:`6039` - remove MPI dependency when only using LCI parcelport
* :hpx-pr:`6038` - Clean up command line handling
* :hpx-pr:`6037` - Avoid performing parcel related background work if networking is disabled
* :hpx-pr:`6036` - Support new datapar backend : SVE
* :hpx-pr:`6035` - Simplify datapar replace copy if
* :hpx-pr:`6034` - Add/Fix documentation of Public API
* :hpx-pr:`6033` - Support for data-parallelism for replace, replace_if, replace_copy, replace_copy_if algorithms
* :hpx-pr:`6032` - Add documentation in public API
* :hpx-pr:`6031` - Expose available cache sizes from topology object
* :hpx-pr:`6030` - Adding parcelport initialization hook for resource partitioner operation
* :hpx-pr:`6029` - Simplify startup code
* :hpx-pr:`6027` - Add/Fix documentation in Public API page
* :hpx-pr:`6026` - add option hpx:force_ipv4 to force resolving hostnames to ipv4 addresses
* :hpx-pr:`6025` - build(docs): remove leftover sections
* :hpx-pr:`6023` - Minor fixes on "How to build on Windows"
* :hpx-pr:`6022` - build(doxy): don't extract private members
* :hpx-pr:`6021` - Adding pu_mask to thread_pool_bulk_scheduler
* :hpx-pr:`6020` - docs: add cppref NamedRequirements support
* :hpx-pr:`6018` - Unseq adaptation for for_each, transform, reduce, transform_reduce, etc.
* :hpx-pr:`6017` - loop and transform_loop unseq adaptation
* :hpx-pr:`6016` - Config and structural updates to support unseq implementation
* :hpx-pr:`6015` - Integrating sync_wait & sync_wait_with_variant
* :hpx-pr:`6012` - docs: add missing links to public api
* :hpx-pr:`6009` - Fixing sender&receiver integration with for_each and for_loop
* :hpx-pr:`6007` - docs: add docs for mutex.hpp
* :hpx-pr:`6006` - Relax future::is_ready where possible
* :hpx-pr:`6005` - reshuffle header tests to different instances
* :hpx-pr:`6004` - Add documentation Public API
* :hpx-pr:`6003` - Always exporting get_component_name implementations
* :hpx-pr:`6002` - Making sure that default constructble arguments are properly constructed during deserialization
* :hpx-pr:`5996` - Add back explicit template parameters to lock_guards for nvcc
* :hpx-pr:`5994` - Fix CTRL+C on windows
* :hpx-pr:`5993` - Using EVE requires C++20
* :hpx-pr:`5992` - This properly terminates an application on Ctrl-C on Windows
* :hpx-pr:`5991` - Support IPV6 on command line for explicit network initialization
* :hpx-pr:`5990` - P2300 enhancements
* :hpx-pr:`5989` - Fix missing documentation in Public API page
* :hpx-pr:`5987` - Attempting to fix timed executor API
* :hpx-pr:`5986` - Fix warnings when building docs 
* :hpx-pr:`5985` - Re-add deprecated tag_policy_tag et.al. types that were removed in V1.8.1
* :hpx-pr:`5981` - docs: add docs for condition_variable.hpp
* :hpx-pr:`5980` - More work on execution::read
* :hpx-pr:`5979` - Remove support for clang-v8 and clang-v9, switch LSU clang-v13 to C++17
* :hpx-pr:`5977` - fix: Compilation errors for -std=c++17 builders
* :hpx-pr:`5975` - docs: fix & improve parallel algorithms documentation 5
* :hpx-pr:`5974` - [P2300] Adapt get completion signatures for awaitable senders
* :hpx-pr:`5973` - defaults boost.context on riscv64
* :hpx-pr:`5972` - Fix documentation for container algorithms
* :hpx-pr:`5971` - added logic to detect riscv compiler configured for 64 bit target
* :hpx-pr:`5968` - adds risc-v 64 bit support
* :hpx-pr:`5967` - Adding missing pieces to sync_wait, adding run_loop
* :hpx-pr:`5966` - docs: fix & improve parallel algorithms documentation 4
* :hpx-pr:`5965` - Fixing inspect problems, adding missing header file
* :hpx-pr:`5962` - Changes in html page of documentation
* :hpx-pr:`5961` - Prevent stalling during shutdown when running hello_world_distributed
* :hpx-pr:`5955` - Fix documentation for container algorithms
* :hpx-pr:`5952` - docs: fix & improve parallel algorithms documentation 3
* :hpx-pr:`5950` - Change executors to directly implement the executor CPOs
* :hpx-pr:`5949` - Converting async combinators into CPOs
* :hpx-pr:`5948` - Adding support for pure sender/receiver based executors to parallel algorithms
* :hpx-pr:`5945` - [P2300] Added fundamental coroutine_traits for S/R
* :hpx-pr:`5883` - Optimization on LCI parcelport: uses LCI_putva
* :hpx-pr:`5872` - Block fork join executor
* :hpx-pr:`5855` - Adding performance test Jenkins builder at LSU
