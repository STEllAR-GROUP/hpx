# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    spell_check_simple
    spell_check_file
   )

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/SpellCheck")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.spell_check.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.spell_check
                              examples.spell_check.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.spell_check.${example_program}
                              ${example_program}_exe)
endforeach()

