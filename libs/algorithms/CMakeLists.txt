# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(algorithms_headers
  hpx/parallel/algorithm.hpp
  hpx/parallel/algorithms/adjacent_difference.hpp
  hpx/parallel/algorithms/adjacent_find.hpp
  hpx/parallel/algorithms/all_any_none.hpp
  hpx/parallel/algorithms/copy.hpp
  hpx/parallel/algorithms/count.hpp
  hpx/parallel/algorithms/destroy.hpp
  hpx/parallel/algorithms/detail/accumulate.hpp
  hpx/parallel/algorithms/detail/dispatch.hpp
  hpx/parallel/algorithms/detail/distance.hpp
  hpx/parallel/algorithms/detail/set_operation.hpp
  hpx/parallel/algorithms/detail/transfer.hpp
  hpx/parallel/algorithms/equal.hpp
  hpx/parallel/algorithms/exclusive_scan.hpp
  hpx/parallel/algorithms/fill.hpp
  hpx/parallel/algorithms/find.hpp
  hpx/parallel/algorithms/for_each.hpp
  hpx/parallel/algorithms/for_loop.hpp
  hpx/parallel/algorithms/for_loop_induction.hpp
  hpx/parallel/algorithms/for_loop_reduction.hpp
  hpx/parallel/algorithms/generate.hpp
  hpx/parallel/algorithms/includes.hpp
  hpx/parallel/algorithms/inclusive_scan.hpp
  hpx/parallel/algorithms/is_heap.hpp
  hpx/parallel/algorithms/is_partitioned.hpp
  hpx/parallel/algorithms/is_sorted.hpp
  hpx/parallel/algorithms/lexicographical_compare.hpp
  hpx/parallel/algorithms/merge.hpp
  hpx/parallel/algorithms/minmax.hpp
  hpx/parallel/algorithms/mismatch.hpp
  hpx/parallel/algorithms/move.hpp
  hpx/parallel/algorithms/partition.hpp
  hpx/parallel/algorithms/reduce_by_key.hpp
  hpx/parallel/algorithms/reduce.hpp
  hpx/parallel/algorithms/remove_copy.hpp
  hpx/parallel/algorithms/remove.hpp
  hpx/parallel/algorithms/replace.hpp
  hpx/parallel/algorithms/reverse.hpp
  hpx/parallel/algorithms/rotate.hpp
  hpx/parallel/algorithms/search.hpp
  hpx/parallel/algorithms/set_difference.hpp
  hpx/parallel/algorithms/set_intersection.hpp
  hpx/parallel/algorithms/set_symmetric_difference.hpp
  hpx/parallel/algorithms/set_union.hpp
  hpx/parallel/algorithms/sort_by_key.hpp
  hpx/parallel/algorithms/sort.hpp
  hpx/parallel/algorithms/swap_ranges.hpp
  hpx/parallel/algorithms/transform_exclusive_scan.hpp
  hpx/parallel/algorithms/transform.hpp
  hpx/parallel/algorithms/transform_inclusive_scan.hpp
  hpx/parallel/algorithms/transform_reduce_binary.hpp
  hpx/parallel/algorithms/transform_reduce.hpp
  hpx/parallel/algorithms/uninitialized_copy.hpp
  hpx/parallel/algorithms/uninitialized_default_construct.hpp
  hpx/parallel/algorithms/uninitialized_fill.hpp
  hpx/parallel/algorithms/uninitialized_move.hpp
  hpx/parallel/algorithms/uninitialized_value_construct.hpp
  hpx/parallel/algorithms/unique.hpp
  hpx/parallel/container_algorithms/all_any_none.hpp
  hpx/parallel/container_algorithms/copy.hpp
  hpx/parallel/container_algorithms/count.hpp
  hpx/parallel/container_algorithms/fill.hpp
  hpx/parallel/container_algorithms/find.hpp
  hpx/parallel/container_algorithms/for_each.hpp
  hpx/parallel/container_algorithms/generate.hpp
  hpx/parallel/container_algorithms.hpp
  hpx/parallel/container_algorithms/is_heap.hpp
  hpx/parallel/container_algorithms/merge.hpp
  hpx/parallel/container_algorithms/minmax.hpp
  hpx/parallel/container_algorithms/move.hpp
  hpx/parallel/container_algorithms/partition.hpp
  hpx/parallel/container_algorithms/remove_copy.hpp
  hpx/parallel/container_algorithms/remove.hpp
  hpx/parallel/container_algorithms/replace.hpp
  hpx/parallel/container_algorithms/reverse.hpp
  hpx/parallel/container_algorithms/rotate.hpp
  hpx/parallel/container_algorithms/search.hpp
  hpx/parallel/container_algorithms/sort.hpp
  hpx/parallel/container_algorithms/transform.hpp
  hpx/parallel/container_algorithms/unique.hpp
  hpx/parallel/datapar.hpp
  hpx/parallel/datapar/iterator_helpers.hpp
  hpx/parallel/datapar/loop.hpp
  hpx/parallel/datapar/transform_loop.hpp
  hpx/parallel/datapar/zip_iterator.hpp
  hpx/parallel/memory.hpp
  hpx/parallel/numeric.hpp
  hpx/parallel/tagspec.hpp
  hpx/parallel/traits/projected.hpp
  hpx/parallel/traits/projected_range.hpp
  hpx/parallel/util/cancellation_token.hpp
  hpx/parallel/util/compare_projected.hpp
  hpx/parallel/util/detail/algorithm_result.hpp
  hpx/parallel/util/detail/chunk_size.hpp
  hpx/parallel/util/detail/chunk_size_iterator.hpp
  hpx/parallel/util/detail/handle_local_exceptions.hpp
  hpx/parallel/util/detail/handle_remote_exceptions.hpp
  hpx/parallel/util/detail/partitioner_iteration.hpp
  hpx/parallel/util/detail/scoped_executor_parameters.hpp
  hpx/parallel/util/detail/select_partitioner.hpp
  hpx/parallel/util/foreach_partitioner.hpp
  hpx/parallel/util/invoke_projected.hpp
  hpx/parallel/util/loop.hpp
  hpx/parallel/util/partitioner.hpp
  hpx/parallel/util/partitioner_with_cleanup.hpp
  hpx/parallel/util/prefetching.hpp
  hpx/parallel/util/projection_identity.hpp
  hpx/parallel/util/scan_partitioner.hpp
  hpx/parallel/util/transfer.hpp
  hpx/parallel/util/transform_loop.hpp
  hpx/parallel/util/zip_iterator.hpp
  hpx/traits/is_value_proxy.hpp
  hpx/traits/segmented_iterator_traits.hpp
)

include(HPX_AddModule)
add_hpx_module(algorithms
  FORCE_LINKING_GEN
  HEADERS ${algorithms_headers}
  # TODO: Really depends on main HPX library, but cannot add that as a
  # dependency yet. Need to have the parts that algorithms depends on in
  # separate modules first.
  DEPENDENCIES
    hpx_assertion
    hpx_concepts
    hpx_concurrency
    hpx_config
    hpx_datastructures
    hpx_execution
    hpx_iterator_support
    hpx_functional
    hpx_statistics
    hpx_synchronization
    hpx_timing
    hpx_type_support
  CMAKE_SUBDIRS examples tests
)

