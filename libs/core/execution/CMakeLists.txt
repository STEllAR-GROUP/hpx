# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(execution_headers
    hpx/execution/algorithms/bulk.hpp
    hpx/execution/algorithms/detach.hpp
    hpx/execution/algorithms/detail/is_negative.hpp
    hpx/execution/algorithms/detail/partial_algorithm.hpp
    hpx/execution/algorithms/detail/predicates.hpp
    hpx/execution/algorithms/detail/single_result.hpp
    hpx/execution/algorithms/ensure_started.hpp
    hpx/execution/algorithms/execute.hpp
    hpx/execution/algorithms/just.hpp
    hpx/execution/algorithms/just_on.hpp
    hpx/execution/algorithms/keep_future.hpp
    hpx/execution/algorithms/let_error.hpp
    hpx/execution/algorithms/let_value.hpp
    hpx/execution/algorithms/make_future.hpp
    hpx/execution/algorithms/on.hpp
    hpx/execution/algorithms/split.hpp
    hpx/execution/algorithms/sync_wait.hpp
    hpx/execution/algorithms/transform.hpp
    hpx/execution/algorithms/when_all.hpp
    hpx/execution/detail/async_launch_policy_dispatch.hpp
    hpx/execution/detail/execution_parameter_callbacks.hpp
    hpx/execution/detail/future_exec.hpp
    hpx/execution/detail/post_policy_dispatch.hpp
    hpx/execution/detail/sync_launch_policy_dispatch.hpp
    hpx/execution/execution.hpp
    hpx/execution/executor_parameters.hpp
    hpx/execution/executors/auto_chunk_size.hpp
    hpx/execution/executors/dynamic_chunk_size.hpp
    hpx/execution/executors/execution.hpp
    hpx/execution/executors/execution_information.hpp
    hpx/execution/executors/execution_parameters.hpp
    hpx/execution/executors/execution_parameters_fwd.hpp
    hpx/execution/executors/fused_bulk_execute.hpp
    hpx/execution/executors/guided_chunk_size.hpp
    hpx/execution/executors/persistent_auto_chunk_size.hpp
    hpx/execution/executors/polymorphic_executor.hpp
    hpx/execution/executors/rebind_executor.hpp
    hpx/execution/executors/static_chunk_size.hpp
    hpx/execution/traits/detail/simd/vector_pack_alignment_size.hpp
    hpx/execution/traits/detail/simd/vector_pack_count_bits.hpp
    hpx/execution/traits/detail/simd/vector_pack_load_store.hpp
    hpx/execution/traits/detail/simd/vector_pack_type.hpp
    hpx/execution/traits/detail/vc/vector_pack_alignment_size.hpp
    hpx/execution/traits/detail/vc/vector_pack_count_bits.hpp
    hpx/execution/traits/detail/vc/vector_pack_load_store.hpp
    hpx/execution/traits/detail/vc/vector_pack_type.hpp
    hpx/execution/traits/executor_traits.hpp
    hpx/execution/traits/future_then_result_exec.hpp
    hpx/execution/traits/is_execution_policy.hpp
    hpx/execution/traits/vector_pack_alignment_size.hpp
    hpx/execution/traits/vector_pack_count_bits.hpp
    hpx/execution/traits/vector_pack_load_store.hpp
    hpx/execution/traits/vector_pack_type.hpp
)

set(execution_sources execution_parameter_callbacks.cpp
                      polymorphic_executor.cpp
)

# cmake-format: off
set(execution_compat_headers
    hpx/async_launch_policy_dispatch.hpp => hpx/execution.hpp
    hpx/parallel/execution.hpp => hpx/execution.hpp
    hpx/parallel/executor_parameters.hpp => hpx/execution.hpp
    hpx/parallel/executors/auto_chunk_size.hpp => hpx/execution.hpp
    hpx/parallel/executors/dynamic_chunk_size.hpp => hpx/execution.hpp
    hpx/parallel/executors/execution.hpp => hpx/execution.hpp
    hpx/parallel/executors/execution_information_fwd.hpp => hpx/execution.hpp
    hpx/parallel/executors/execution_information.hpp => hpx/execution.hpp
    hpx/parallel/executors/execution_parameters_fwd.hpp => hpx/execution.hpp
    hpx/parallel/executors/execution_parameters.hpp => hpx/execution.hpp
    hpx/parallel/executors/fused_bulk_execute.hpp => hpx/execution.hpp
    hpx/parallel/executors/guided_chunk_size.hpp => hpx/execution.hpp
    hpx/parallel/executors/persistent_auto_chunk_size.hpp => hpx/execution.hpp
    hpx/parallel/executors/post_policy_dispatch.hpp => hpx/execution.hpp
    hpx/parallel/executors/rebind_executor.hpp => hpx/execution.hpp
    hpx/parallel/executors/static_chunk_size.hpp => hpx/execution.hpp
    hpx/parallel/traits/vector_pack_alignment_size.hpp => hpx/execution.hpp
    hpx/parallel/traits/vector_pack_count_bits.hpp => hpx/execution.hpp
    hpx/parallel/traits/vector_pack_load_store.hpp => hpx/execution.hpp
    hpx/parallel/traits/vector_pack_type.hpp => hpx/execution.hpp
    hpx/sync_launch_policy_dispatch.hpp => hpx/execution.hpp
    hpx/traits/executor_traits.hpp => hpx/execution.hpp
    hpx/traits/is_execution_policy.hpp => hpx/execution.hpp
)
# cmake-format: on

if(TARGET Vc::vc)
  set(execution_optional_dependencies Vc::vc)
endif()

include(HPXLocal_AddModule)
hpx_local_add_module(
  local execution
  SOURCES ${execution_sources}
  HEADERS ${execution_headers}
  COMPAT_HEADERS ${execution_compat_headers}
  DEPENDENCIES ${execution_optional_dependencies}
  MODULE_DEPENDENCIES
    hpx_async_base
    hpx_async_combinators
    hpx_config_local
    hpx_threading
    hpx_pack_traversal
    hpx_errors
    hpx_memory
    hpx_thread_support
    hpx_timing
    hpx_topology
  CMAKE_SUBDIRS examples tests
)
