# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Threading tests
set(threading_tests
    condition_variable2
    condition_variable3
    condition_variable4
    error_callback
    jthread1
    jthread2
    stack_check
    start_stop_callbacks
    stop_tokencb1
    stop_tokenrace
    thread
    thread_affinity
    thread_id
    thread_launching
    thread_mf
    thread_stacksize
    thread_yield
)

if(HPX_WITH_THREAD_STACKOVERFLOW_DETECTION)
  set(threading_tests ${threading_tests} thread_stacksize_overflow)
  set(threading_tests ${threading_tests} thread_stacksize_overflow_v2)
endif()

if(HPX_WITH_THREAD_LOCAL_STORAGE)
  set(threading_tests ${threading_tests} tss)
endif()

set(condition_variable2_PARAMETERS THREADS_PER_LOCALITY 4)
set(condition_variable3_PARAMETERS THREADS_PER_LOCALITY 4)
set(condition_variable4_PARAMETERS THREADS_PER_LOCALITY 4)
set(jthread1_PARAMETERS THREADS_PER_LOCALITY 4)
set(jthread2_PARAMETERS THREADS_PER_LOCALITY 4)
set(stop_tokencb1_PARAMETERS THREADS_PER_LOCALITY 4)
set(stop_tokenrace_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_affinity_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_affinity_LIBRARIES DEPENDENCIES PRIVATE Hwloc::hwloc)
set(thread_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_id_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_launching_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_mf_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_stacksize_PARAMETERS LOCALITIES 2)
set(tss_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${threading_tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(${test}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Threading")

  add_hpx_unit_test("modules.threading" ${test})

endforeach()

if(HPX_WITH_THREAD_STACKOVERFLOW_DETECTION)
  set_tests_properties(tests.unit.threads.thread_stacksize_overflow PROPERTIES
    PASS_REGULAR_EXPRESSION "Stack overflow in coroutine at address 0x[0-9a-fA-F]*")

  set_tests_properties(tests.unit.threads.thread_stacksize_overflow_v2 PROPERTIES
    PASS_REGULAR_EXPRESSION "Stack overflow in coroutine at address 0x[0-9a-fA-F]*")
endif()
