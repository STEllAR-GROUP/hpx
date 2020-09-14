# Copyright (c) 2017 Taeguk Kwon
# Copyright (c) 2017 Google
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
    benchmark_inplace_merge
    benchmark_is_heap
    benchmark_is_heap_until
    benchmark_merge
    benchmark_partition
    benchmark_partition_copy
    benchmark_remove
    benchmark_remove_if
    benchmark_unique
    benchmark_unique_copy
)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(benchmarks ${benchmarks} transform_reduce_scaling)
  set(transform_reduce_scaling_FLAGS DEPENDENCIES iostreams_component)
endif()

foreach(benchmark ${benchmarks})
  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL ${${benchmark}_FLAGS}
    FOLDER "Benchmarks/Modules/Parallelism/Algorithms"
  )

  add_hpx_performance_test(
    "modules.algorithms" ${benchmark} ${${benchmark}_PARAMETERS}
  )
endforeach()
