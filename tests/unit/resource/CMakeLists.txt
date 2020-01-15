# Copyright (c) 2017-2020 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    scheduler_binding_check
    scheduler_priority_check
    shutdown_suspended_pus
    suspend_disabled
    suspend_pool
    suspend_pool_external
    suspend_runtime
    suspend_thread
    suspend_thread_external
    suspend_thread_timed
  )

if(HPX_WITH_SHARED_PRIORITY_SCHEDULER)
  set(tests ${tests}
      cross_pool_injection
    )
  set(cross_pool_injection_PARAMETERS THREADS_PER_LOCALITY -1)
endif()

# NB. threads = -2 = threads = 'cores'
# NB. threads = -1 = threads = 'all'

set(scheduler_priority_check_PARAMETERS THREADS_PER_LOCALITY -1)
set(scheduler_binding_check_PARAMETERS THREADS_PER_LOCALITY -1)
set(shutdown_suspended_pus_PARAMETERS THREADS_PER_LOCALITY 4)
set(suspend_disabled_PARAMETERS THREADS_PER_LOCALITY 4)
set(suspend_pool_PARAMETERS THREADS_PER_LOCALITY 4)
set(suspend_pool_external_PARAMETERS THREADS_PER_LOCALITY 4)
set(suspend_runtime_PARAMETERS THREADS_PER_LOCALITY 4)
set(suspend_thread_PARAMETERS THREADS_PER_LOCALITY 4)
set(suspend_thread_external_PARAMETERS THREADS_PER_LOCALITY 4)
set(suspend_thread_timed_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/ResourcePartitioner/")

  add_hpx_unit_test("resource" ${test} ${${test}_PARAMETERS})

endforeach()

