# Copyright (c) 2020 John Biddiscombe
# Copyright (c) 2020 Weile Wei
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks libcds_hazard_pointer_overhead)

foreach(benchmark ${benchmarks})

  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add benchmark executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL ${${benchmark}_FLAGS}
    FOLDER "Benchmarks/Modules/LibCDS"
    DEPENDENCIES hpx_libcds
  )

  target_include_directories(${benchmark}_test PRIVATE ${LIBCDS_SOURCE_DIR})

  # add a custom target for this benchmark
  add_hpx_performance_test(
    "modules.libcds" ${benchmark} ${${benchmark}_PARAMETERS}
  )

endforeach()
