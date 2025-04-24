# Copyright (c) 2020 The STE||AR-Group
#               2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests runtime_type parcel_pool shutdown_suspended_thread
          start_stop_callbacks
)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests ${tests} handled_exception unhandled_exception)
endif()

set(unhandled_exception_PARAMETERS FAILURE_EXPECTED)

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS ${${test}_PARAMETERS} THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Full/InitRuntime")

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("modules.init_runtime" ${test} ${${test}_PARAMETERS})
endforeach()
