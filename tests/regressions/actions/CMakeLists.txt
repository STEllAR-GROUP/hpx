# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(components)

set(tests
    async_deferred_1523
    component_action_move_semantics
    make_continuation_1615
    plain_action_1330
    plain_action_1550
    plain_action_move_semantics
    return_future_2847
   )

set(plain_action_1330_FLAGS
    DEPENDENCIES iostreams_component)

set(plain_action_move_semantics_FLAGS
    DEPENDENCIES action_move_semantics_component)
set(plain_action_move_semantics_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 1)

set(component_action_move_semantics_FLAGS
    DEPENDENCIES action_move_semantics_component)
set(component_action_move_semantics_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 1)

set(return_future_2847_PARAMETERS LOCALITIES 2)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Regressions/Actions")

  add_hpx_regression_test("actions" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.actions_dir.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.actions_dir
                              tests.regressions.actions_dir.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.actions_dir.${test}
                              ${test}_test_exe)
endforeach()

