# Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
#                    LLC (NTESS).
# Copyright (c) 2019 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks
    async_replicate_distributed
    async_replicate_distributed_validate
    async_replicate_distributed_vote
    async_replicate_distributed_vote_validate
    plain_async_replicate_distributed
)

foreach(benchmark ${benchmarks})

  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add benchmark executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL ${${benchmark}_FLAGS}
    FOLDER "Benchmarks/Modules/Resiliency/Replicate"
  )

  # add a custom target for this benchmark
  add_hpx_performance_test(
    "modules.resiliency.replicate" ${benchmark} ${${benchmark}_PARAMETERS}
  )

endforeach()
