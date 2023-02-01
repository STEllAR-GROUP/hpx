..
    Copyright (C) 2007-2022 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_1_9_0:

===========================
|hpx| V1.9.0 (TBD)
===========================

General changes
===============
- Added RISC-V 64bit support. HPX is now compatible with RISC-V architectures which have revolutionized the HPC world.
- LCI parcelport has been optimized to transfer parcels with fewer messages and use the HPX resource partitioner for 
  its progress thread allocation. It should generally provide better performance than before. It also removes its 
  dependency on the MPI library.
- HPX dependency on Boost was further relaxed by replacing headers from Boost.Range, Boost.Tokenizer and Boost.Lockfree.
- Improvements took place on our parallel algorithms implementation.
- Our Senders/Receivers (P2300) integration was extended:

  - Coroutines were integrated with senders/receivers. ``get_completion_signatures`` now works with awaitable senders.
  - ``with_awaitable_senders`` allows the passed senders to retrieve the value i.e. senders are transparently
    awaitable from within a coroutine.
  - ``when_all_vector`` was added.

- ``sync_wait`` and ``sync_wait_with_variant`` sender consumers were added. The user can now initiate the execution of
  their asynchronous pipeline by blocking the current thread that executes the main() function until the result
  is retrieved.
- The combinators for futures (a.k.a. async_combinators) ``when_*``, ``wait_*``, ``wait_*_nothrow`` were turned into CPOs 
  allowing for end-user customization. For more information on the async_combinators refer to the documentation,
  https://hpx-docs.stellar-group.org/latest/html/libs/core/async_combinators/docs/index.html?highlight=combinators.
- The new datapar backend SVE allows simd and par_simd execution policies to exploit dataparalleism in the processors that 
  have SVE vector registers like A64FX and Neoverse V1.
- The documentation for parallel algorithms, container algorithms was further improved. The Public API page was vastly enriched. 
- Copy button shortkey was added at the top-right of code-blocks.
- Pragma directive that reports warnings as errors on MSVC was fixed. 
- Command line argument ``--hpx:loopback_network`` was added to facilitie debugging with networks.

Breaking changes
================

- Stopped supporting Clang V8, the minimal version supported is now Clang V10.
- Stopped supporting gcc V8, the minimal version supported is now gcc V9.
- Stopped supporting Visual Studio 2015, the minimal version supported is
  now Visual Studio 2019.
- ``tag_policy_tag`` et.al. were re-added after HPX V1.8.1 depracation.
- ``get_chunk_size`` and ``processing_units_count`` API is now expecting the time for one iteration as an argument.
- ``hpx::parallel::reduce_by_key`` is deprecated in favor of ``hpx::experimental::reduce_by_key``.
- ``hpx::parallel::task_block`` is deprecated in favor of ``hpx::experimental::task_block``.
- ``hpx::util::mem_fn`` is deprecated in favor of ``hpx::mem_fn``.
- ``hpx::util::invoke``                           is deprecated in favor of ``hpx::invoke``.
- ``hpx::util::invoke_r``                         is deprecated in favor of ``hpx::invoke_r``.
- ``hpx::util::invoke_fused``                     is deprecated in favor of ``hpx::invoke_fused``.
- ``hpx::util::invoke_fused_r``                   is deprecated in favor of ``hpx::invoke_fused_r``.
- ``hpx::util::unlock_guard``                     is deprecated in favor of ``hpx::unlock_guard``.
- ``hpx::parallel::v1::reduce_by_key``            is deprecated in favor of ``hpx::experimental::reduce_by_key``.
- ``hpx::parallel::v1::sort_by_key``              is deprecated in favor of ``hpx::experimental::sort_by_key``.
- ``hpx::parallel::task_canceled_exception``      is deprecated in favor of ``hpx::experimental::task_canceled_exception``.
- ``hpx::parallel::task_block``                   is deprecated in favor of ``hpx::experimental::task_block``.
- ``hpx::parallel::define_task_block``            is deprecated in favor of ``hpx::experimental::define_task_block``.  
- ``hpx::parallel::define_task_block_restore_thread`` is deprecated in favor of ``hpx::experimental::define_task_block_restore_thread``.
- ``hpx::execution::experimental::task_group`` is deprecated in favor of        ``hpx::experimental::task_group``.


Closed issues 
=============

* :hpx-issue:`6108` - cxx17_aligned_new.cpp on msvc fails due to wrong pragma directive
* :hpx-issue:`6045` - Can't call nullary callables wrapped with `hpx::unwrapping`
* :hpx-issue:`6008` - Missing `constexpr` default constructor for `hpx::mutex`
* :hpx-issue:`5999` - Add HPX Conda package to conda-forge
* :hpx-issue:`5998` - Serializing multiple arguments when applying distributed action results in segfault
* :hpx-issue:`5908` - Windows: duplicated symbols in static builds
* :hpx-issue:`5802` - Lost status is_ready from future
* :hpx-issue:`5767` - Performance drop on Piz Daint
* :hpx-issue:`5752` - Implement stride_view from P1899 (experimental)
* :hpx-issue:`5744` - HPX_WITH_FETCH_ASIO not working on Ookami
* :hpx-issue:`5561` - Possible race condition in helper thread / hpx::cout

Closed pull requests
====================

* :hpx-pr:`6132` - Fixing to_non_par() for parallel simd policies
* :hpx-pr:`6130` - Remove the mutex lock in the critical path of get_partitioner.
* :hpx-pr:`6127` - Working around gccV9 problem that prevent us from storing enum classes in bit fields
* :hpx-pr:`6126` - Deprecate hpx::parallel::task_block in favor of hpx::experimental::ta…
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
* :hpx-pr:`6026` - add option hpx:force_ipv4 to force resolving hostnames to ipv4 adresses
* :hpx-pr:`6025` - build(docs): remove leftover sections
* :hpx-pr:`6023` - Minor fixes on "How to build on Windows"
* :hpx-pr:`6022` - build(doxy): don't extract private members
* :hpx-pr:`6021` - Adding pu_mask to thread_pool_bulk_scheduler
* :hpx-pr:`6020` - docs: add cppref NamedRequirements support
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
* :hpx-pr:`5979` - Unsupport clang-v8 and clang-v9, switch LSU clang-v13 to C++17
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
