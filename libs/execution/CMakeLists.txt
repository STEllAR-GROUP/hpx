# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(execution_headers
    hpx/execution/algorithms/detail/is_negative.hpp
    hpx/execution/algorithms/detail/predicates.hpp
    hpx/execution/detail/async_launch_policy_dispatch.hpp
    hpx/execution/detail/future_exec.hpp
    hpx/execution/detail/execution_parameter_callbacks.hpp
    hpx/execution/detail/post_policy_dispatch.hpp
    hpx/execution/detail/sync_launch_policy_dispatch.hpp
    hpx/execution/execution.hpp
    hpx/execution/executor_parameters.hpp
    hpx/execution/executors/auto_chunk_size.hpp
    hpx/execution/executors/dynamic_chunk_size.hpp
    hpx/execution/executors/execution.hpp
    hpx/execution/executors/execution_information_fwd.hpp
    hpx/execution/executors/execution_information.hpp
    hpx/execution/executors/execution_parameters_fwd.hpp
    hpx/execution/executors/execution_parameters.hpp
    hpx/execution/executors/fused_bulk_execute.hpp
    hpx/execution/executors/guided_chunk_size.hpp
    hpx/execution/executors/persistent_auto_chunk_size.hpp
    hpx/execution/executors/polymorphic_executor.hpp
    hpx/execution/executors/rebind_executor.hpp
    hpx/execution/executors/static_chunk_size.hpp
    hpx/execution/traits/detail/vc/vector_pack_alignment_size.hpp
    hpx/execution/traits/detail/vc/vector_pack_count_bits.hpp
    hpx/execution/traits/detail/vc/vector_pack_load_store.hpp
    hpx/execution/traits/detail/vc/vector_pack_type.hpp
    hpx/execution/traits/executor_traits.hpp
    hpx/execution/traits/future_then_result_exec.hpp
    hpx/execution/traits/is_execution_policy.hpp
    hpx/execution/traits/is_executor.hpp
    hpx/execution/traits/is_executor_parameters.hpp
    hpx/execution/traits/vector_pack_alignment_size.hpp
    hpx/execution/traits/vector_pack_count_bits.hpp
    hpx/execution/traits/vector_pack_load_store.hpp
    hpx/execution/traits/vector_pack_type.hpp
)

set(execution_sources execution_parameter_callbacks.cpp
                      polymorphic_executor.cpp
)

set(execution_compat_headers
    hpx/async_launch_policy_dispatch.hpp
    hpx/parallel/execution.hpp
    hpx/parallel/executor_parameters.hpp
    hpx/parallel/executors/auto_chunk_size.hpp
    hpx/parallel/executors/dynamic_chunk_size.hpp
    hpx/parallel/executors/execution.hpp
    hpx/parallel/executors/execution_information_fwd.hpp
    hpx/parallel/executors/execution_information.hpp
    hpx/parallel/executors/execution_parameters_fwd.hpp
    hpx/parallel/executors/execution_parameters.hpp
    hpx/parallel/executors/fused_bulk_execute.hpp
    hpx/parallel/executors/guided_chunk_size.hpp
    hpx/parallel/executors/persistent_auto_chunk_size.hpp
    hpx/parallel/executors/post_policy_dispatch.hpp
    hpx/parallel/executors/rebind_executor.hpp
    hpx/parallel/executors/static_chunk_size.hpp
    hpx/parallel/traits/vector_pack_alignment_size.hpp
    hpx/parallel/traits/vector_pack_count_bits.hpp
    hpx/parallel/traits/vector_pack_load_store.hpp
    hpx/parallel/traits/vector_pack_type.hpp
    hpx/sync_launch_policy_dispatch.hpp
    hpx/traits/executor_traits.hpp
    hpx/traits/is_execution_policy.hpp
    hpx/traits/is_executor.hpp
    hpx/traits/is_executor_parameters.hpp
)

include(HPX_AddModule)
add_hpx_module(
  execution
  SOURCES ${execution_sources}
  HEADERS ${execution_headers}
  DEPRECATION_WARNINGS
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  COMPAT_HEADERS ${execution_compat_headers}
  DEPENDENCIES
    hpx_assertion
    hpx_async_base
    hpx_async_combinators
    hpx_allocator_support
    hpx_concepts
    hpx_config
    hpx_errors
    hpx_datastructures
    hpx_errors
    hpx_functional
    hpx_iterator_support
    hpx_memory
    hpx_preprocessor
    hpx_resource_partitioner
    hpx_synchronization
    hpx_threading
    hpx_threadmanager
    hpx_thread_support
    hpx_timing
    hpx_topology
    hpx_type_support
    hpx_pack_traversal
  CMAKE_SUBDIRS examples tests
)
