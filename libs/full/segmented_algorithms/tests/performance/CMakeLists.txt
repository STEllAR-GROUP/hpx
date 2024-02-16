# Copyright (c) 2016 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks minmax_element_performance)

if (HPX_WITH_NANOBENCH_BENCHMARKS)
  set(benchmarks ${benchmarks} minmax_nanobench)
endif()

foreach(benchmark ${benchmarks})
  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  set(deps iostreams_component partitioned_vector_component)
  if (HPX_WITH_NANOBENCH_BENCHMARKS)
    set(deps ${deps} nanobench)
  endif()

  # add example executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${benchmark}_FLAGS}
    DEPENDENCIES ${deps}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Benchmarks/Modules/Full/SegmentedAlgorithms"
  )

  add_hpx_performance_test(
    "modules.segmented_algorithms" ${benchmark} ${${benchmark}_PARAMETERS}
  )
endforeach()
