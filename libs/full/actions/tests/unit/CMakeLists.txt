# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests set_thread_state thread_affinity thread_stacksize)

if(HPX_WITH_NETWORKING)
  set(tests ${tests} serialize_buffer zero_copy_serialization)
endif()

set(set_thread_state_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_stacksize_PARAMETERS LOCALITIES 2)
set(thread_affinity_PARAMETERS THREADS_PER_LOCALITY 4 RUN_SERIAL)
set(thread_affinity_LIBRARIES DEPENDENCIES PRIVATE Hwloc::hwloc)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS} ${${test}_LIBRARIES}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Full/Actions"
  )

  add_hpx_unit_test("modules.actions" ${test} ${${test}_PARAMETERS})
endforeach()
