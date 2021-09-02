# Copyright (c) 2007-2019 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    composable_guard
    customize_async
    disable_thread_stealing_executor
    enumerate_threads
    event_synchronization
    executor_with_thread_hooks
    fibonacci_await
    fibonacci_dataflow
    fibonacci_futures
    fibonacci_local
    file_serialization
    hello_world
    interval_timer
    latch_local
    local_channel
    local_channel_docs
    pipeline1
    potpourri
    safe_object
    shared_mutex
    simple_future_continuation
    timed_futures
    timed_wake
    use_main_thread
    vector_counting_dotproduct
    vector_zip_dotproduct
    wait_composition
)

# TODO: These examples currently fail. Disabled until they are fixed.
set(disabled_tests fibonacci_await)

set(customize_async_PARAMETERS THREADS_PER_LOCALITY 4)
set(enumerate_threads_PARAMETERS THREADS_PER_LOCALITY 4)
set(event_synchronization_PARAMETERS THREADS_PER_LOCALITY 4)
set(hello_world_PARAMETERS THREADS_PER_LOCALITY 4)
set(interval_timer_PARAMETERS THREADS_PER_LOCALITY 4)
set(local_channel_PARAMETERS THREADS_PER_LOCALITY 4)
set(simple_future_continuation_PARAMETERS THREADS_PER_LOCALITY 4)
set(timed_futures_PARAMETERS THREADS_PER_LOCALITY 4)
set(vector_counting_dotproduct_PARAMETERS THREADS_PER_LOCALITY 4)
set(vector_zip_dotproduct_PARAMETERS THREADS_PER_LOCALITY 4)

# the following example fails compiling on someversions of MSVC
if(HPXLocal_WITH_TUPLE_RVALUE_SWAP AND (NOT MSVC))
  list(APPEND example_programs sort_by_key_demo)
endif()

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  hpx_local_add_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Quickstart"
  )

  target_link_libraries(
    ${example_program} PRIVATE ${${example_program}_LIBRARIES}
  )
  hpx_local_add_example_target_dependencies("quickstart" ${example_program})

  if(HPXLocal_WITH_TESTS
     AND HPXLocal_WITH_TESTS_EXAMPLES
     AND NOT ("${example_program}" IN_LIST disabled_tests)
  )
    hpx_local_add_example_test(
      "quickstart" ${example_program} ${${example_program}_PARAMETERS}
    )
  endif()

endforeach()
