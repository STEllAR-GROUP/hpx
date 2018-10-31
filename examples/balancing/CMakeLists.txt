# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    os_thread_num
    hpx_thread_phase)

set(os_thread_num_FLAGS DEPENDENCIES iostreams_component)

set(os_thread_num_PARAMETERS THREADS_PER_LOCALITY 4)
set(hpx_thread_phase_PARAMETERS THREADS_PER_LOCALITY 4)

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

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test("balancing" ${example_program} ${${example_program}_PARAMETERS})
    add_hpx_pseudo_target(tests.examples.balancing.${example_program})
    add_hpx_pseudo_dependencies(tests.examples.balancing
                                tests.examples.balancing.${example_program})
    add_hpx_pseudo_dependencies(tests.examples.balancing.${example_program}
                                ${example_program}_exe)
  endif()
endforeach()
