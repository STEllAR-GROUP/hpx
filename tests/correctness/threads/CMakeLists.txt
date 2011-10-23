# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    set_thread_state
   )

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}
                     SOURCES ${sources}
                     ${${test}_FLAGS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.correctness.threads.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.correctness.threads
                              tests.correctness.threads.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.correctness.threads.${test}
                              ${test}_exe)
endforeach()

