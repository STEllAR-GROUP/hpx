# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    component_ctors
    component_in_executable
    component_inheritance
    error_handling
    event_synchronization 
    factorial
    fibonacci
    fibonacci_one
    fibonacci_futures
    fractals
    fractals_struct
    #fractals_oclm
    interest_calculator
    non_atomic_rma
    pingpong
    quicksort
    simplest_hello_world
    timed_wake
    use_main_thread
    wait_composition
   )

if(HPX_HAVE_CXX11_LAMBDAS)
  set(example_programs ${example_programs}
      hello_world
      1d_wave_equation
     )
endif()

set(component_in_executable_FLAGS DEPENDENCIES iostreams_component)
set(component_ctors_FLAGS DEPENDENCIES iostreams_component)
set(event_synchronization_FLAGS DEPENDENCIES iostreams_component)

set(error_handling_FLAGS DEPENDENCIES iostreams_component)
if(HPX_USE_COMPRESSION_SNAPPY AND SNAPPY_FOUND)
  set(hello_world_FLAGS DEPENDENCIES iostreams_component compress_snappy_lib)
else()
  set(hello_world_FLAGS DEPENDENCIES iostreams_component)
endif()
set(1d_wave_equation_FLAGS DEPENDENCIES iostreams_component)
set(simplest_hello_world_FLAGS DEPENDENCIES iostreams_component)
set(component_inheritance_FLAGS DEPENDENCIES iostreams_component)
set(interest_calculator_FLAGS DEPENDENCIES dataflow_component)

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/Quickstart/${example_program}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.quickstart.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.quickstart
                              examples.quickstart.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.quickstart.${example_program}
                              ${example_program}_exe)
endforeach()

