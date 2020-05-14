# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests contiguous_index_queue lockfree_fifo)

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
    hpx_type_support
)

set(contiguous_index_queue_PARAMETERS THREADS_PER_LOCALITY 4)

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

target_compile_definitions(
  lockfree_fifo_test PRIVATE HPX_MODULE_STATIC_LINKING HPX_NO_VERSION_CHECK
)
target_include_directories(lockfree_fifo_test PRIVATE ${HPX_SOURCE_DIR})
target_link_libraries(lockfree_fifo_test PRIVATE hpx_allocator_support)
