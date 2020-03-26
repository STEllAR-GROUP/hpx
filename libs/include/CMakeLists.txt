# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(include_headers
  hpx/include/actions.hpp
  hpx/include/agas.hpp
  hpx/include/applier.hpp
  hpx/include/apply.hpp
  hpx/include/async.hpp
  hpx/include/bind.hpp
  hpx/include/client.hpp
  hpx/include/components.hpp
  hpx/include/compression.hpp
  hpx/include/compression_registration.hpp
  hpx/include/compute.hpp
  hpx/include/dataflow.hpp
  hpx/include/datapar.hpp
  hpx/include/future.hpp
  hpx/include/lcos.hpp
  hpx/include/local_lcos.hpp
  hpx/include/naming.hpp
  hpx/include/parallel_adjacent_difference.hpp
  hpx/include/parallel_adjacent_find.hpp
  hpx/include/parallel_algorithm.hpp
  hpx/include/parallel_all_any_none_of.hpp
  hpx/include/parallel_container_algorithm.hpp
  hpx/include/parallel_copy.hpp
  hpx/include/parallel_count.hpp
  hpx/include/parallel_destroy.hpp
  hpx/include/parallel_equal.hpp
  hpx/include/parallel_exception_list.hpp
  hpx/include/parallel_execution.hpp
  hpx/include/parallel_execution_policy.hpp
  hpx/include/parallel_executor_information.hpp
  hpx/include/parallel_executor_parameters.hpp
  hpx/include/parallel_executors.hpp
  hpx/include/parallel_fill.hpp
  hpx/include/parallel_find.hpp
  hpx/include/parallel_for_each.hpp
  hpx/include/parallel_for_loop.hpp
  hpx/include/parallel_generate.hpp
  hpx/include/parallel_is_heap.hpp
  hpx/include/parallel_is_partitioned.hpp
  hpx/include/parallel_is_sorted.hpp
  hpx/include/parallel_lexicographical_compare.hpp
  hpx/include/parallel_memory.hpp
  hpx/include/parallel_merge.hpp
  hpx/include/parallel_minmax.hpp
  hpx/include/parallel_mismatch.hpp
  hpx/include/parallel_move.hpp
  hpx/include/parallel_numeric.hpp
  hpx/include/parallel_partition.hpp
  hpx/include/parallel_reduce.hpp
  hpx/include/parallel_remove_copy.hpp
  hpx/include/parallel_remove.hpp
  hpx/include/parallel_replace.hpp
  hpx/include/parallel_reverse.hpp
  hpx/include/parallel_rotate.hpp
  hpx/include/parallel_scan.hpp
  hpx/include/parallel_search.hpp
  hpx/include/parallel_set_operations.hpp
  hpx/include/parallel_sort.hpp
  hpx/include/parallel_swap_ranges.hpp
  hpx/include/parallel_task_block.hpp
  hpx/include/parallel_transform.hpp
  hpx/include/parallel_transform_reduce.hpp
  hpx/include/parallel_transform_scan.hpp
  hpx/include/parallel_uninitialized_copy.hpp
  hpx/include/parallel_uninitialized_default_construct.hpp
  hpx/include/parallel_uninitialized_fill.hpp
  hpx/include/parallel_uninitialized_move.hpp
  hpx/include/parallel_uninitialized_value_construct.hpp
  hpx/include/parallel_unique.hpp
  hpx/include/parcel_coalescing.hpp
  hpx/include/parcelset.hpp
  hpx/include/performance_counters.hpp
  hpx/include/plain_actions.hpp
  hpx/include/resource_partitioner.hpp
  hpx/include/run_as.hpp
  hpx/include/runtime.hpp
  hpx/include/serialization.hpp
  hpx/include/sync.hpp
  hpx/include/thread_executors.hpp
  hpx/include/threadmanager.hpp
  hpx/include/threads.hpp
  hpx/include/traits.hpp
  hpx/include/util.hpp)

include(HPX_AddModule)
add_hpx_module(include
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  HEADERS ${include_headers}
  DEPENDENCIES
    hpx_algorithms
    hpx_local_lcos
    hpx_naming_base
    hpx_performance_counters
    hpx_resource_partitioner
    hpx_segmented_algorithms
    hpx_serialization
    hpx_threadmanager
    hpx_threading
    hpx_util
  CMAKE_SUBDIRS tests
)
