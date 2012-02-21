# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    action_move_semantics
   )

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Regressions/Actions")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.actions.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.actions
                              tests.regressions.actions.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.actions.${test}
                              ${test}_test_exe)
endforeach()

