# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

set(segmented_algorithms_headers
    hpx/parallel/segmented_algorithm.hpp
    hpx/parallel/segmented_algorithms/adjacent_difference.hpp
    hpx/parallel/segmented_algorithms/adjacent_find.hpp
    hpx/parallel/segmented_algorithms/all_any_none.hpp
    hpx/parallel/segmented_algorithms/count.hpp
    hpx/parallel/segmented_algorithms/detail/dispatch.hpp
    hpx/parallel/segmented_algorithms/detail/reduce.hpp
    hpx/parallel/segmented_algorithms/detail/scan.hpp
    hpx/parallel/segmented_algorithms/detail/transfer.hpp
    hpx/parallel/segmented_algorithms/exclusive_scan.hpp
    hpx/parallel/segmented_algorithms/fill.hpp
    hpx/parallel/segmented_algorithms/find.hpp
    hpx/parallel/segmented_algorithms/for_each.hpp
    hpx/parallel/segmented_algorithms/generate.hpp
    hpx/parallel/segmented_algorithms/inclusive_scan.hpp
    hpx/parallel/segmented_algorithms/minmax.hpp
    hpx/parallel/segmented_algorithms/reduce.hpp
    hpx/parallel/segmented_algorithms/traits/zip_iterator.hpp
    hpx/parallel/segmented_algorithms/transform_exclusive_scan.hpp
    hpx/parallel/segmented_algorithms/transform.hpp
    hpx/parallel/segmented_algorithms/transform_inclusive_scan.hpp
    hpx/parallel/segmented_algorithms/transform_reduce.hpp
)

set(segmented_algorithms_compat_headers hpx/util/zip_iterator.hpp)

include(HPX_AddModule)
add_hpx_module(
  full segmented_algorithms
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  HEADERS ${segmented_algorithms_headers}
  COMPAT_HEADERS ${segmented_algorithms_compat_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  CMAKE_SUBDIRS examples tests
)
