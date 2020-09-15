# Copyright (c) 2019 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks channel_mpmc_throughput channel_mpsc_throughput
               channel_spsc_throughput
)

set(channel_mpmc_throughput_PARAMETERS THREADS_PER_LOCALITY 2)
set(channel_mpsc_throughput_PARAMETERS THREADS_PER_LOCALITY 2)
set(channel_spsc_throughputs_PARAMETERS THREADS_PER_LOCALITY 2)

foreach(benchmark ${benchmarks})

  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add benchmark executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL ${${benchmark}_FLAGS}
    FOLDER "Benchmarks/Modules/Core/Synchronization"
  )

  # add a custom target for this benchmark
  add_hpx_performance_test(
    "modules.synchronization" ${benchmark} ${${benchmark}_PARAMETERS}
  )

endforeach()
