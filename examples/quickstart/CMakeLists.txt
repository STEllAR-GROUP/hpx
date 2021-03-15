# Copyright (c) 2007-2019 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    composable_guard
    disable_thread_stealing_executor
    executor_with_thread_hooks
    fibonacci_await
    fibonacci_dataflow
    fibonacci_futures
    fibonacci_local
    file_serialization
    latch_local
    pipeline1
    safe_object
    timed_wake
    use_main_thread
    wait_composition
)

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
  set(example_programs ${example_programs} init_globally)
endif()

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(example_programs
      ${example_programs}
      1d_wave_equation
      allow_unknown_options
      channel_docs
      command_line_handling
      component_ctors
      component_in_executable
      component_inheritance
      component_with_custom_heap
      component_with_executor
      customize_async
      data_actions
      enumerate_threads
      error_handling
      event_synchronization
      factorial
      fibonacci
      fibonacci_futures_distributed
      fibonacci_one
      hello_world_1
      hello_world_2
      hello_world_distributed
      interest_calculator
      interval_timer
      invoke_function
      latch_remote
      local_channel_docs
      local_channel
      partitioned_vector_spmd_foreach
      pingpong
      potpourri
      print_to_console
      receive_buffer
      sierpinski
      simple_future_continuation
      timed_futures
      vector_counting_dotproduct
      vector_zip_dotproduct
      zerocopy_rdma
  )
endif()

# TODO: These examples currently fail. Disabled until they are fixed.
set(disabled_tests error_handling fibonacci_await non_atomic_rma shared_mutex
                   zerocopy_rdma
)

set(1d_wave_equation_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(allow_unknown_options_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(channel_docs_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(command_line_handling_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(component_in_executable_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(component_ctors_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(component_inheritance_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(component_with_custom_heap_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(component_with_executor_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(customize_async_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(enumerate_threads_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(event_synchronization_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(error_handling_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(hello_world_1_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(hello_world_2_FLAGS COMPONENT_DEPENDENCIES iostreams)
if(HPX_HAVE_COMPRESSION_SNAPPY AND SNAPPY_FOUND)
  set(hello_world_distributed_FLAGS COMPONENT_DEPENDENCIES iostreams
                                    DEPENDENCIES compress_snappy
  )
else()
  set(hello_world_distributed_FLAGS COMPONENT_DEPENDENCIES iostreams)
endif()
set(interval_timer_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(local_channel_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(local_channel_docs_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(sierpinski_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(partitioned_vector_spmd_foreach_FLAGS COMPONENT_DEPENDENCIES iostreams
                                          partitioned_vector
)
set(pingpong_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(potpourri_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(shared_mutex_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(simple_future_continuation_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(timed_futures_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(vector_counting_dotproduct_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(vector_zip_dotproduct_FLAGS COMPONENT_DEPENDENCIES iostreams)

set(1d_wave_equation_PARAMETERS THREADS_PER_LOCALITY 4
                                "-Ihpx.stacks.use_guard_pages!=0"
)
set(allow_unknown_options_PARAMETERS THREADS_PER_LOCALITY 4)
set(command_line_handling_PARAMETERS THREADS_PER_LOCALITY 4)
set(component_in_executable_PARAMETERS THREADS_PER_LOCALITY 4)
set(component_ctors_PARAMETERS THREADS_PER_LOCALITY 4)
set(component_inheritance_PARAMETERS THREADS_PER_LOCALITY 4)
set(component_with_executor_PARAMETERS THREADS_PER_LOCALITY 4)
set(customize_async_PARAMETERS THREADS_PER_LOCALITY 4)
set(enumerate_threads_PARAMETERS THREADS_PER_LOCALITY 4)
set(event_synchronization_PARAMETERS THREADS_PER_LOCALITY 4)
set(error_handling_PARAMETERS THREADS_PER_LOCALITY 4)
set(hello_world_1_PARAMETERS THREADS_PER_LOCALITY 4)
set(hello_world_2_PARAMETERS THREADS_PER_LOCALITY 4)
set(hello_world_distributed_PARAMETERS THREADS_PER_LOCALITY 4)
set(interval_timer_PARAMETERS THREADS_PER_LOCALITY 4)
set(local_channel_PARAMETERS THREADS_PER_LOCALITY 4)
set(sierpinski_PARAMETERS THREADS_PER_LOCALITY 4)
set(partitioned_vector_spmd_foreach_PARAMETERS THREADS_PER_LOCALITY 4)
set(pingpong_PARAMETERS THREADS_PER_LOCALITY 4)
set(simple_future_continuation_PARAMETERS THREADS_PER_LOCALITY 4)
set(timed_futures_PARAMETERS THREADS_PER_LOCALITY 4)
set(vector_counting_dotproduct_PARAMETERS THREADS_PER_LOCALITY 4)
set(vector_zip_dotproduct_PARAMETERS THREADS_PER_LOCALITY 4)

# the following example fails compiling on someversions of MSVC
if(HPX_WITH_TUPLE_RVALUE_SWAP
   AND (NOT MSVC)
   AND HPX_WITH_DISTRIBUTED_RUNTIME
)
  set(example_programs ${example_programs} sort_by_key_demo)
  set(sort_by_key_demo_FLAGS COMPONENT_DEPENDENCIES iostreams)
endif()

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Quickstart"
  )

  target_link_libraries(
    ${example_program} PRIVATE ${${example_program}_LIBRARIES}
  )
  add_hpx_example_target_dependencies("quickstart" ${example_program})

  if(HPX_WITH_TESTS
     AND HPX_WITH_TESTS_EXAMPLES
     AND NOT ("${example_program}" IN_LIST disabled_tests)
  )
    add_hpx_example_test(
      "quickstart" ${example_program} ${${example_program}_PARAMETERS}
    )
  endif()

endforeach()
