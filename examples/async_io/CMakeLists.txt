# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    async_io_simple
    async_io_action
    async_io_external
    async_io_low_level)

foreach(example_program ${example_programs})

  set(${example_program}_FLAGS DEPENDENCIES iostreams_component)

  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/AsyncIO/${example_program}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.async_io.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.async_io
                              examples.async_io.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.async_io.${example_program}
                              ${example_program}_exe)
endforeach()

