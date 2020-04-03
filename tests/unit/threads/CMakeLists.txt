# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    lockfree_fifo
    schedule_last
    set_thread_state
    stack_check
    error_callback
    start_stop_callbacks
    thread
    thread_affinity
    thread_id
    thread_launching
    thread_mf
    thread_stacksize
    thread_yield
   )

if(HPX_WITH_EMBEDDED_THREAD_POOLS_COMPATIBILITY)
  set(tests ${tests}
    resource_manager
    thread_suspension_executor)
endif()

if(HPX_WITH_THREAD_STACKOVERFLOW_DETECTION)
  set(tests ${tests} thread_stacksize_overflow)
  set(tests ${tests} thread_stacksize_overflow_v2)
endif()

if(HPX_WITH_THREAD_LOCAL_STORAGE)
  set(tests ${tests} tss)
endif()

set(lockfree_fifo_FLAGS NOLIBS)
set(lockfree_fifo_LIBRARIES
  DEPENDENCIES
    hpx_dependencies_boost
    hpx_assertion
    hpx_config
    hpx_concurrency
    hpx_datastructures
    hpx_functional
    hpx_program_options
    hpx_testing
    hpx_type_support)

set(resource_manager_PARAMETERS THREADS_PER_LOCALITY 4)

set(set_thread_state_PARAMETERS THREADS_PER_LOCALITY 4)

set(thread_affinity_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_affinity_LIBRARIES DEPENDENCIES PRIVATE Hwloc::hwloc)

set(thread_PARAMETERS THREADS_PER_LOCALITY 4)

set(thread_id_PARAMETERS THREADS_PER_LOCALITY 4)

set(thread_launching_PARAMETERS THREADS_PER_LOCALITY 4)

set(thread_mf_PARAMETERS THREADS_PER_LOCALITY 4)

set(thread_stacksize_PARAMETERS LOCALITIES 2)

set(tss_PARAMETERS THREADS_PER_LOCALITY 4)

###############################################################################
foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    ${${test}_FLAGS}
    ${${test}_LIBRARIES}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Threads")

  add_hpx_unit_test("threads" ${test} ${${test}_PARAMETERS})

endforeach()

target_compile_definitions(lockfree_fifo_test
  PRIVATE HPX_MODULE_STATIC_LINKING HPX_NO_VERSION_CHECK)
target_include_directories(lockfree_fifo_test
  PRIVATE ${HPX_SOURCE_DIR})

if(HPX_WITH_THREAD_STACKOVERFLOW_DETECTION)
  set_tests_properties(tests.unit.threads.thread_stacksize_overflow PROPERTIES
    PASS_REGULAR_EXPRESSION "Stack overflow in coroutine at address 0x[0-9a-fA-F]*")

  set_tests_properties(tests.unit.threads.thread_stacksize_overflow_v2 PROPERTIES
    PASS_REGULAR_EXPRESSION "Stack overflow in coroutine at address 0x[0-9a-fA-F]*")
endif()
