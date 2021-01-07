# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    after_588
    call_promise_get_gid_more_than_once
    exception_from_continuation_1613
    lifetime_588
    lifetime_588_1
    promise_leak_996
    promise_1620
    receive_buffer_1733
    set_hpx_limit_798
    shared_stated_leaked_1211
    sliding_semaphore_2338
    wait_for_1751
    wait_for_action_2796
)

set(promise_1620_FLAGS DEPENDENCIES iostreams_component)
set(after_588_PARAMETERS LOCALITIES 2)
set(exception_from_continuation_1613_PARAMETERS THREADS_PER_LOCALITY 2)
set(lifetime_588_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)
set(lifetime_588_1_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)
set(receive_buffer_1733_PARAMETERS LOCALITIES 2)
set(safely_destroy_cv_1481_PARAMETERS THREADS_PER_LOCALITY 2)
set(sliding_semaphore_2338_PARAMETERS THREADS_PER_LOCALITY 2)
set(wait_for_1751_PARAMETERS THREADS_PER_LOCALITY 4)

if(HPX_WITH_THREAD_EXECUTORS_COMPATIBILITY)
  list(APPEND tests shared_future_then_with_executor_3634)
endif()

# Create test cases
foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/LCOs"
  )

  target_link_libraries(${test}_test PRIVATE ${${test}_LIBRARIES})
  add_hpx_regression_test("lcos" ${test} ${${test}_PARAMETERS})

endforeach()

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  add_hpx_regression_test(
    "lcos"
    after_588_4
    EXECUTABLE
    after_588
    PSEUDO_DEPS_NAME
    after_588
    LOCALITIES
    2
    THREADS_PER_LOCALITY
    4
  )
endif()
