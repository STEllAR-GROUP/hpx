# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(executors_headers
  hpx/parallel/algorithms/detail/is_negative.hpp
  hpx/parallel/algorithms/detail/predicates.hpp
  hpx/parallel/datapar/execution_policy_fwd.hpp
  hpx/parallel/datapar/execution_policy.hpp
  hpx/parallel/exception_list.hpp
  hpx/parallel/execution.hpp
  hpx/parallel/execution_policy_fwd.hpp
  hpx/parallel/execution_policy.hpp
  hpx/parallel/executor_parameters.hpp
  hpx/parallel/executors/auto_chunk_size.hpp
  hpx/parallel/executors/default_executor.hpp
  hpx/parallel/executors/distribution_policy_executor.hpp
  hpx/parallel/executors/dynamic_chunk_size.hpp
  hpx/parallel/executors/execution_fwd.hpp
  hpx/parallel/executors/execution.hpp
  hpx/parallel/executors/execution_information_fwd.hpp
  hpx/parallel/executors/execution_information.hpp
  hpx/parallel/executors/execution_parameters_fwd.hpp
  hpx/parallel/executors/execution_parameters.hpp
  hpx/parallel/executors/fused_bulk_execute.hpp
  hpx/parallel/executors/guided_chunk_size.hpp
  hpx/parallel/executors.hpp
  hpx/parallel/executors/parallel_executor_aggregated.hpp
  hpx/parallel/executors/parallel_executor.hpp
  hpx/parallel/executors/persistent_auto_chunk_size.hpp
  hpx/parallel/executors/pool_executor.hpp
  hpx/parallel/executors/post_policy_dispatch.hpp
  hpx/parallel/executors/rebind_executor.hpp
  hpx/parallel/executors/sequenced_executor.hpp
  hpx/parallel/executors/service_executors.hpp
  hpx/parallel/executors/static_chunk_size.hpp
  hpx/parallel/executors/this_thread_executors.hpp
  hpx/parallel/executors/thread_execution.hpp
  hpx/parallel/executors/thread_execution_information.hpp
  hpx/parallel/executors/thread_pool_attached_executors.hpp
  hpx/parallel/executors/thread_pool_executors.hpp
  hpx/parallel/executors/thread_pool_os_executors.hpp
  hpx/parallel/executors/thread_timed_execution.hpp
  hpx/parallel/executors/timed_execution_fwd.hpp
  hpx/parallel/executors/timed_execution.hpp
  hpx/parallel/executors/timed_executors.hpp
  hpx/parallel/traits/detail/boost_simd/vector_pack_alignment_size.hpp
  hpx/parallel/traits/detail/boost_simd/vector_pack_count_bits.hpp
  hpx/parallel/traits/detail/boost_simd/vector_pack_load_store.hpp
  hpx/parallel/traits/detail/boost_simd/vector_pack_type.hpp
  hpx/parallel/traits/detail/vc/vector_pack_alignment_size.hpp
  hpx/parallel/traits/detail/vc/vector_pack_count_bits.hpp
  hpx/parallel/traits/detail/vc/vector_pack_load_store.hpp
  hpx/parallel/traits/detail/vc/vector_pack_type.hpp
  hpx/parallel/traits/vector_pack_alignment_size.hpp
  hpx/parallel/traits/vector_pack_count_bits.hpp
  hpx/parallel/traits/vector_pack_load_store.hpp
  hpx/parallel/traits/vector_pack_type.hpp
  hpx/traits/executor_traits.hpp
  hpx/traits/is_execution_policy.hpp
  hpx/traits/is_executor.hpp
  hpx/traits/is_executor_parameters.hpp
  hpx/traits/is_timed_executor.hpp
)

include(HPX_AddModule)
add_hpx_module(executors
    FORCE_LINKING_GEN
    HEADERS ${executors_headers}
    DEPENDENCIES hpx_assertion hpx_config
    CMAKE_SUBDIRS examples tests
)
