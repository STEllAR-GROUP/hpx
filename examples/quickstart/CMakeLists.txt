# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    factorial
    fibonacci
    non_atomic_rma
    quicksort
    timed_wake
    simplest_hello_world
   )

if(HPX_CXX11_LAMBDAS)
  set(example_programs ${example_programs}
      hello_world
     )
endif()

set(hello_world_FLAGS DEPENDENCIES iostreams_component)

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS})

  # add a custom target for this example
  add_hpx_pseudo_target(examples.quickstart.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.quickstart
                              examples.quickstart.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.quickstart.${example_program}
                              ${example_program}_exe)
endforeach()

