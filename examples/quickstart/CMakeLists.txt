# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    component_ctors
    component_in_executable
    component_inheritance
    component_inheritance_simple
    component_with_executor
    composable_guard
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
    interest_calculator
    non_atomic_rma
    pingpong
    print_to_console
    quicksort
    simple_future_continuation
    simplest_hello_world
    timed_futures
    timed_wake
    use_main_thread
    wait_composition
    zerocopy_rdma
   )

if(HPX_LOCAL_SCHEDULER OR HPX_ALL_SCHEDULERS)
  set(example_programs ${example_programs}
      fractals_executor
     )
endif()

if(HPX_HAVE_CXX11_LAMBDAS)
  set(example_programs ${example_programs}
      1d_wave_equation
      hello_world
      fibonacci_dataflow
      pipeline1
      vector_zip_dotproduct
      vector_counting_dotproduct
     )
endif()

set(component_in_executable_FLAGS DEPENDENCIES iostreams_component)
set(component_ctors_FLAGS DEPENDENCIES iostreams_component)
set(component_with_executor_FLAGS DEPENDENCIES iostreams_component)
set(event_synchronization_FLAGS DEPENDENCIES iostreams_component)

set(error_handling_FLAGS DEPENDENCIES iostreams_component)
set(fractals_FLAGS DEPENDENCIES iostreams_component)
set(fractals_executor_FLAGS DEPENDENCIES iostreams_component)
if(HPX_HAVE_COMPRESSION_SNAPPY AND SNAPPY_FOUND)
  set(hello_world_FLAGS DEPENDENCIES iostreams_component compress_snappy_lib)
else()
  set(hello_world_FLAGS DEPENDENCIES iostreams_component)
endif()
set(1d_wave_equation_FLAGS DEPENDENCIES iostreams_component)
set(simplest_hello_world_FLAGS DEPENDENCIES iostreams_component)
set(component_inheritance_FLAGS DEPENDENCIES iostreams_component)
set(component_inheritance_simple_FLAGS DEPENDENCIES iostreams_component)
set(interest_calculator_FLAGS DEPENDENCIES dataflow_component)
set(simple_future_continuation_FLAGS DEPENDENCIES iostreams_component)
set(timed_futures_FLAGS DEPENDENCIES iostreams_component)
set(pingpong_FLAGS DEPENDENCIES iostreams_component)
set(vector_zip_dotproduct_FLAGS DEPENDENCIES iostreams_component)
set(vector_counting_dotproduct_FLAGS DEPENDENCIES iostreams_component)

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

