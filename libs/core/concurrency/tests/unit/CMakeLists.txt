# Copyright (c) 2011 Bryce Adelstein-Lelbach
# Copyright (c) 2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    contiguous_index_queue
    freelist
    lockfree_fifo
    non_contiguous_index_queue
    queue
    queue_stress
    stack
    stack_destructor
    stack_stress
    tagged_ptr
)

set(contiguous_index_queue_PARAMETERS THREADS_PER_LOCALITY 4)
set(non_contiguous_index_queue_PARAMETERS THREADS_PER_LOCALITY 4)
set(freelist_PARAMETERS THREADS_PER_LOCALITY 4)
set(queue_stress_PARAMETERS THREADS_PER_LOCALITY 4)
set(stack_stress_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS} ${${test}_LIBRARIES}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Core/Concurrency"
  )

  add_hpx_unit_test("modules.concurrency" ${test} ${${test}_PARAMETERS})
endforeach()
