# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(group
    timed)

set(example_programs
    timed_wake)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     MODULE ${group}
                     SOURCES ${sources})

  # add a custom target for this example 
  add_hpx_pseudo_target(examples.${group}.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.${group}
                              examples.${group}.${example_program})

  # add dependencies to pseudo-target 
  add_hpx_pseudo_dependencies(examples.${group}.${example_program}
                              ${example_program}_exe)
endforeach() 

