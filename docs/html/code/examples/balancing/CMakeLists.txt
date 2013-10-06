# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    os_thread_num
    px_thread_phase)

set(os_thread_num_FLAGS DEPENDENCIES iostreams_component)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/Balancing/${example_program}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.balancing.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.balancing
                              examples.balancing.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.balancing.${example_program}
                              ${example_program}_exe)
endforeach()

