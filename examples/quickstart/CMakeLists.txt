# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    basic_example
    distributed_test
    distributed_amrtest
    non_atomic_rma
    factorial
    fibonacci
    error_handling1
    error_handling2
    error_handling3
    error_handling4
    timed_wake
    quicksort
   )

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     MODULE quickstart
                     SOURCES ${sources})

  # add a custom target for this example 
  add_hpx_pseudo_target(examples.quickstart.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.quickstart
                              examples.quickstart.${example_program})

  # add dependencies to pseudo-target 
  add_hpx_pseudo_dependencies(examples.quickstart.${example_program}
                              ${example_program}_exe)
endforeach() 

if("${HPX_AGAS_VERSION}" STREQUAL "2")
  set(example_programs
      hello_world
     )
  
  foreach(example_program ${example_programs})
    set(sources ${example_program}.cpp)
  
    source_group("Source Files" FILES ${sources})
  
    # add example executable
    add_hpx_executable(${example_program}
                       MODULE quickstart
                       DEPENDENCIES iostreams_component
                       SOURCES ${sources})
  
    # add a custom target for this example 
    add_hpx_pseudo_target(examples.quickstart.${example_program})
  
    # make pseudo-targets depend on master pseudo-target
    add_hpx_pseudo_dependencies(examples.quickstart
                                examples.quickstart.${example_program})
  
    # add dependencies to pseudo-target 
    add_hpx_pseudo_dependencies(examples.quickstart.${example_program}
                                ${example_program}_exe)
  endforeach() 
endif()
 
