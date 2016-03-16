# Copyright (c) 2007-2015 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    allow_unknown_options
    component_ctors
    component_in_executable
    component_inheritance
    component_with_executor
    composable_guard
    customize_async
    data_actions
    error_handling
    event_synchronization
    factorial
    fibonacci
    fibonacci_await
    fibonacci_one
    fibonacci_futures
    fibonacci_futures_distributed
    fractals
    fractals_struct
    init_globally
    sierpinski
    interest_calculator
    interval_timer
    latch_local
    latch_remote
    non_atomic_rma
    partitioned_vector_spmd_foreach
    pingpong
    print_to_console
    quicksort
    safe_object
    shared_mutex
    simple_future_continuation
    simplest_hello_world
    timed_futures
    timed_wake
    use_main_thread
    wait_composition
    zerocopy_rdma
   )

set(allow_unknown_options_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(component_in_executable_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(component_ctors_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(component_inheritance_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(component_with_executor_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(customize_async_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(event_synchronization_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(error_handling_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(fractals_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(interval_timer_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(sierpinski_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(partitioned_vector_spmd_foreach_FLAGS
    COMPONENT_DEPENDENCIES iostreams partitioned_vector)
set(pingpong_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(shared_mutex_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(simplest_hello_world_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(simple_future_continuation_FLAGS COMPONENT_DEPENDENCIES iostreams)
set(timed_futures_FLAGS COMPONENT_DEPENDENCIES iostreams)

if(HPX_HAVE_LOCAL_SCHEDULER OR HPX_HAVE_ALL_SCHEDULERS)
  set(example_programs ${example_programs}
      fractals_executor
     )
  set(fractals_executor_FLAGS COMPONENT_DEPENDENCIES iostreams)
endif()

if(HPX_WITH_CXX11_LAMBDAS)
  set(example_programs ${example_programs}
      1d_wave_equation
      hello_world
      fibonacci_dataflow
      pipeline1
      vector_counting_dotproduct
      vector_zip_dotproduct
     )

  set(1d_wave_equation_FLAGS COMPONENT_DEPENDENCIES iostreams)
  if(HPX_HAVE_COMPRESSION_SNAPPY AND SNAPPY_FOUND)
    set(hello_world_FLAGS COMPONENT_DEPENDENCIES iostreams DEPENDENCIES compress_snappy_lib)
  else()
    set(hello_world_FLAGS COMPONENT_DEPENDENCIES iostreams)
  endif()
  set(vector_counting_dotproduct_FLAGS COMPONENT_DEPENDENCIES iostreams)
  set(vector_zip_dotproduct_FLAGS COMPONENT_DEPENDENCIES iostreams)
endif()

if(HPX_WITH_TUPLE_RVALUE_SWAP)
  set(example_programs ${example_programs}
      sort_by_key_demo
     )
  set(sort_by_key_demo_FLAGS COMPONENT_DEPENDENCIES iostreams)
endif()

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/Quickstart")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.quickstart.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.quickstart
                              examples.quickstart.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.quickstart.${example_program}
                              ${example_program}_exe)
endforeach()

