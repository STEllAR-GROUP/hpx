# Copyright (c) 2016 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests)

set(tests ${tests}
    default_executor
    for_each_compute
    for_loop_compute
    transform_compute
   )
set(default_executor_CUDA On)
set(for_each_compute_CUDA On)
set(for_loop_compute_CUDA On)
set(transform_compute_CUDA On)

foreach(test ${tests})
  if(${${test}_CUDA})
    set(sources ${test}.cu)
  else()
    set(sources ${test}.cpp)
  endif()

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Compute/CUDA"
  )

  add_hpx_unit_test("modules.compute_cuda" ${test} ${${test}_PARAMETERS})
endforeach()
