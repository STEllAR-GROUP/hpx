# Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
#                    LLC (NTESS).
# Copyright (c) 2019 Nikunj Gupta
# Copyright (c) 2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
    1d_stencil
    1d_stencil_replay
    1d_stencil_checksum
    async_replay
    async_replay_validate
    dataflow_replay
    dataflow_replay_validate
    pure_async_for_replay
    pure_dataflow
)

foreach(benchmark ${benchmarks})

  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add benchmark executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL ${${benchmark}_FLAGS}
    FOLDER "Benchmarks/Modules/Parallelism/Resiliency/Replay"
  )

  # add a custom target for this benchmark
  add_hpx_performance_test(
    "modules.resiliency.replay" ${benchmark} ${${benchmark}_PARAMETERS}
  )

endforeach()
