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
      async_replicate_distributed
      async_replicate_distributed_validate
      async_replicate_distributed_vote
      async_replicate_distributed_vote_validate
      pure_async_for_replicate_distributed
  )

  set(async_replicate_distributed_PARAMETERS LOCALITIES 2)
  set(async_replicate_distributed_validate_PARAMETERS LOCALITIES 2)
  set(async_replicate_distributed_vote_PARAMETERS LOCALITIES 2)
  set(async_replicate_distributed_vote_validate_PARAMETERS LOCALITIES 2)
  set(pure_async_for_replicate_distributed_PARAMETERS LOCALITIES 2)
endif()

foreach(benchmark ${benchmarks})

  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add benchmark executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL ${${benchmark}_FLAGS}
    FOLDER "Benchmarks/Modules/Full/Resiliency/Replicate"
  )

  # add a custom target for this benchmark
  add_hpx_performance_test(
    "modules.resiliency_distributed.replicate" ${benchmark}
    ${${benchmark}_PARAMETERS}
  )

endforeach()
