# Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
#                    LLC (NTESS).
# Copyright (c) 2019 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks)

if(HPX_WITH_NETWORKING)
  set(benchmarks
      1d_stencil_distributed 1d_stencil_replay_distributed
      async_replay_distributed async_replay_distributed_validate
      plain_async_distributed
  )

  set(1d_stencil_distributed_PARAMETERS LOCALITIES 2)
  set(1d_stencil_replay_distributed_PARAMETERS LOCALITIES 2)
  set(async_replay_distributed_PARAMETERS LOCALITIES 2)
  set(async_replay_distributed_validate_PARAMETERS LOCALITIES 2)
  set(plain_async_distributed_PARAMETERS LOCALITIES 2)
endif()

foreach(benchmark ${benchmarks})

  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add benchmark executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL ${${benchmark}_FLAGS}
    FOLDER "Benchmarks/Modules/Full/Resiliency/Replay"
  )

  # add a custom target for this benchmark
  add_hpx_performance_test(
    "modules.resiliency_distributed.replay" ${benchmark}
    ${${benchmark}_PARAMETERS}
  )

endforeach()
