# Copyright (c) 2011 Bryce Adelstein-Lelbach
# Copyright (c) 2012 Daniel Kogler
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks osu_bibw osu_bw osu_latency osu_multi_lat)

foreach(benchmark ${benchmarks})

  set(sources osu_base.cpp ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${benchmark}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Benchmarks/Modules/Full/Collectives/OSU"
  )

  add_hpx_performance_test(
    "modules.collectives" ${benchmark} ${${benchmark}_PARAMETERS}
  )
endforeach()

set(coll_benchmarks # osu_bcast
    # osu_scatter
)

foreach(benchmark ${coll_benchmarks})
  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${benchmark}_FLAGS} EXCLUDE_FROM_ALL
    FOLDER "Benchmarks/Modules/Full/Collectives/OSU"
  )

  add_hpx_performance_test(
    "modules.collectives" ${benchmark} ${${benchmark}_PARAMETERS}
  )

endforeach()
