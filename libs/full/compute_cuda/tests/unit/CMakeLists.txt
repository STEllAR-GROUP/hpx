# Copyright (c) 2016 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests)

set(tests ${tests} default_executor for_each_compute for_loop_compute)

# Note: transform_compute is still broken with cuda 11
if(HPX_WITH_CLANG_CUDA OR HPX_WITH_HIP)
  set(tests ${tests} transform_compute)
endif()

foreach(test ${tests})
  set(sources ${test}.cu)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Full/Compute/CUDA"
  )

  add_hpx_unit_test(
    "modules.compute_cuda" ${test} ${${test}_PARAMETERS} RUN_SERIAL
  )
endforeach()
