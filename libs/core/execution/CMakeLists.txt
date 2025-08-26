# Copyright (c) 2019-2024 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(execution_headers
    hpx/execution/algorithms/as_sender.hpp
    hpx/execution/algorithms/bulk.hpp
    hpx/execution/algorithms/detail/is_negative.hpp
    hpx/execution/algorithms/detail/inject_scheduler.hpp
    hpx/execution/algorithms/detail/partial_algorithm.hpp
    hpx/execution/algorithms/detail/predicates.hpp
    hpx/execution/algorithms/detail/single_result.hpp
    hpx/execution/algorithms/ensure_started.hpp
    hpx/execution/algorithms/execute.hpp
    hpx/execution/algorithms/just.hpp
    hpx/execution/algorithms/keep_future.hpp
    hpx/execution/algorithms/let_error.hpp
    hpx/execution/algorithms/let_stopped.hpp
    hpx/execution/algorithms/let_value.hpp
    hpx/execution/algorithms/make_future.hpp
    hpx/execution/algorithms/run_loop.hpp
    hpx/execution/algorithms/schedule_from.hpp
    hpx/execution/algorithms/split.hpp
    hpx/execution/algorithms/start_detached.hpp
    hpx/execution/algorithms/sync_wait.hpp
    hpx/execution/algorithms/then.hpp
    hpx/execution/algorithms/transfer.hpp
    hpx/execution/algorithms/transfer_just.hpp
    hpx/execution/algorithms/when_all.hpp
    hpx/execution/algorithms/when_all_vector.hpp
    hpx/execution/detail/async_launch_policy_dispatch.hpp
    hpx/execution/detail/execution_parameter_callbacks.hpp
    hpx/execution/detail/future_exec.hpp
    hpx/execution/detail/post_policy_dispatch.hpp
    hpx/execution/detail/sync_launch_policy_dispatch.hpp
    hpx/execution/execution.hpp
    hpx/execution/executor_parameters.hpp
    hpx/execution/executors/adaptive_static_chunk_size.hpp
    hpx/execution/executors/auto_chunk_size.hpp
    hpx/execution/executors/collect_chunking_parameters.hpp
    hpx/execution/executors/default_parameters.hpp
    hpx/execution/executors/dynamic_chunk_size.hpp
    hpx/execution/executors/execution.hpp
    hpx/execution/executors/execution_information.hpp
    hpx/execution/executors/execution_parameters.hpp
    hpx/execution/executors/execution_parameters_fwd.hpp
    hpx/execution/executors/fused_bulk_execute.hpp
    hpx/execution/executors/guided_chunk_size.hpp
    hpx/execution/executors/max_num_chunks.hpp
    hpx/execution/executors/num_cores.hpp
    hpx/execution/executors/persistent_auto_chunk_size.hpp
    hpx/execution/executors/polymorphic_executor.hpp
    hpx/execution/executors/rebind_executor.hpp
    hpx/execution/executors/static_chunk_size.hpp
    hpx/execution/queries/get_allocator.hpp
    hpx/execution/queries/get_scheduler.hpp
    hpx/execution/queries/get_delegatee_scheduler.hpp
    hpx/execution/queries/get_stop_token.hpp
    hpx/execution/queries/read.hpp
    hpx/execution/traits/detail/eve/vector_pack_alignment_size.hpp
    hpx/execution/traits/detail/eve/vector_pack_all_any_none.hpp
    hpx/execution/traits/detail/eve/vector_pack_conditionals.hpp
    hpx/execution/traits/detail/eve/vector_pack_count_bits.hpp
    hpx/execution/traits/detail/eve/vector_pack_find.hpp
    hpx/execution/traits/detail/eve/vector_pack_get_set.hpp
    hpx/execution/traits/detail/eve/vector_pack_load_store.hpp
    hpx/execution/traits/detail/eve/vector_pack_reduce.hpp
    hpx/execution/traits/detail/eve/vector_pack_type.hpp
    hpx/execution/traits/detail/simd/vector_pack_alignment_size.hpp
    hpx/execution/traits/detail/simd/vector_pack_all_any_none.hpp
    hpx/execution/traits/detail/simd/vector_pack_conditionals.hpp
    hpx/execution/traits/detail/simd/vector_pack_count_bits.hpp
    hpx/execution/traits/detail/simd/vector_pack_find.hpp
    hpx/execution/traits/detail/simd/vector_pack_get_set.hpp
    hpx/execution/traits/detail/simd/vector_pack_load_store.hpp
    hpx/execution/traits/detail/simd/vector_pack_reduce.hpp
    hpx/execution/traits/detail/simd/vector_pack_simd.hpp
    hpx/execution/traits/detail/simd/vector_pack_type.hpp
    hpx/execution/traits/detail/vc/vector_pack_alignment_size.hpp
    hpx/execution/traits/detail/vc/vector_pack_all_any_none.hpp
    hpx/execution/traits/detail/vc/vector_pack_conditionals.hpp
    hpx/execution/traits/detail/vc/vector_pack_count_bits.hpp
    hpx/execution/traits/detail/vc/vector_pack_find.hpp
    hpx/execution/traits/detail/vc/vector_pack_get_set.hpp
    hpx/execution/traits/detail/vc/vector_pack_load_store.hpp
    hpx/execution/traits/detail/vc/vector_pack_reduce.hpp
    hpx/execution/traits/detail/vc/vector_pack_type.hpp
    hpx/execution/traits/executor_traits.hpp
    hpx/execution/traits/future_then_result_exec.hpp
    hpx/execution/traits/is_execution_policy.hpp
    hpx/execution/traits/vector_pack_alignment_size.hpp
    hpx/execution/traits/vector_pack_all_any_none.hpp
    hpx/execution/traits/vector_pack_conditionals.hpp
    hpx/execution/traits/vector_pack_count_bits.hpp
    hpx/execution/traits/vector_pack_find.hpp
    hpx/execution/traits/vector_pack_get_set.hpp
    hpx/execution/traits/vector_pack_load_store.hpp
    hpx/execution/traits/vector_pack_reduce.hpp
    hpx/execution/traits/vector_pack_type.hpp
)

set(execution_sources execution_parameter_callbacks.cpp
                      polymorphic_executor.cpp run_loop.cpp
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
    hpx/parallel/traits/vector_pack_all_any_none.hpp => hpx/execution.hpp
    hpx/parallel/traits/vector_pack_conditionals.hpp => hpx/execution.hpp
    hpx/parallel/traits/vector_pack_count_bits.hpp => hpx/execution.hpp
    hpx/parallel/traits/vector_pack_find.hpp => hpx/execution.hpp
    hpx/parallel/traits/vector_pack_get_set.hpp => hpx/execution.hpp
    hpx/parallel/traits/vector_pack_load_store.hpp => hpx/execution.hpp
    hpx/parallel/traits/vector_pack_load_reduce.hpp => hpx/execution.hpp
    hpx/parallel/traits/vector_pack_type.hpp => hpx/execution.hpp
    hpx/sync_launch_policy_dispatch.hpp => hpx/execution.hpp
    hpx/traits/executor_traits.hpp => hpx/execution.hpp
    hpx/traits/is_execution_policy.hpp => hpx/execution.hpp
)
# cmake-format: on

if(TARGET Vc::vc)
  set(execution_optional_dependencies Vc::vc)
endif()

if(TARGET eve::eve)
  set(execution_optional_dependencies eve::eve)
endif()

if(TARGET SVE::sve)
  set(execution_optional_dependencies SVE::sve)
endif()

include(HPX_AddModule)
add_hpx_module(
  core execution
  SOURCES ${execution_sources}
  HEADERS ${execution_headers}
  COMPAT_HEADERS ${execution_compat_headers}
  DEPENDENCIES ${execution_optional_dependencies}
  MODULE_DEPENDENCIES
    hpx_async_base
    hpx_async_combinators
    hpx_concurrency
    hpx_config
    hpx_threading
    hpx_pack_traversal
    hpx_errors
    hpx_lock_registration
    hpx_memory
    hpx_synchronization
    hpx_thread_support
    hpx_timing
    hpx_topology
    hpx_type_support
  CMAKE_SUBDIRS examples tests
)
