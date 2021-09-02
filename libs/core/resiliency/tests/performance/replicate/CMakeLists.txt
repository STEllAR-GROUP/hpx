# Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
#                    LLC (NTESS).
# Copyright (c) 2019 Nikunj Gupta
# Copyright (c) 2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks async_replicate async_replicate_validate async_replicate_vote
               async_replicate_vote_validate pure_async_for_replicate
)

# The Intel compiler version 19.1.1.217 does not manage to compile these. Other
# versions have not been tested.
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  list(APPEND benchmarks 1d_stencil_replicate 1d_stencil_replicate_checksum)
endif()

foreach(benchmark ${benchmarks})

  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add benchmark executable
  hpx_local_add_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL ${${benchmark}_FLAGS}
    FOLDER "Benchmarks/Modules/Core/Resiliency/Replicate"
  )

  # add a custom target for this benchmark
  hpx_local_add_performance_test(
    "modules.resiliency.replicate" ${benchmark} ${${benchmark}_PARAMETERS}
  )

endforeach()
