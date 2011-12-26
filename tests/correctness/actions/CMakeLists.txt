# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    pass_by_const_reference
   )

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Correctness/Actions")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.correctness.actions.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.correctness.actions
                              tests.correctness.actions.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.correctness.actions.${test}
                              ${test}_exe)
endforeach()

