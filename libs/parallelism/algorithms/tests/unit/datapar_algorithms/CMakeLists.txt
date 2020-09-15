# Copyright (c) 2014-2019 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests)

if(HPX_WITH_DATAPAR_VC)
  set(tests
      ${tests}
      count_datapar
      countif_datapar
      foreach_datapar
      foreach_datapar_zipiter
      foreachn_datapar
      transform_datapar
      transform_binary_datapar
      transform_binary2_datapar
      transform_reduce_binary_datapar
  )
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Parallelism/Algorithms/Datapar"
  )

  add_hpx_unit_test("modules.algorithms" ${test} ${${test}_PARAMETERS})
endforeach()
