# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    non_const_ref_action
   )

add_hpx_component(non_const_ref_action_component
  FOLDER "Tests/Unit/Actions"
  SOURCES components/non_const_ref_action_component.cpp
  HEADERS components/non_const_ref_action_component.hpp)

add_hpx_pseudo_dependencies(tests.regressions.actions action_move_semantics)

set(non_const_ref_action_FLAGS
    DEPENDENCIES non_const_ref_action_component_component)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Unit/Actions")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.actions.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.actions
                              tests.unit.actions.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.actions.${test}
                              ${test}_test_exe)
endforeach()
