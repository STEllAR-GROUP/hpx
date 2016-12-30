# Copyright (c) 2007-2017 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(sine)

set(example_programs
    access_counter_set
    simplest_performance_counter
   )

set(access_counter_set_FLAGS COMPONENT_DEPENDENCIES iostreams)

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  set(${example_program}_FLAGS DEPENDENCIES iostreams_component)

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/PerformanceCounters/${example_program}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.performance_counters.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.performance_counters
                              examples.performance_counters.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.performance_counters.${example_program}
                              ${example_program}_exe)
endforeach()


