# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    assert_fail
    assert_succeed
)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     NOHPX_INIT NOLIBS
                     DEPENDENCIES hpx_assertion
                     EXCLUDE_FROM_ALL
                     FOLDER "Tests/Unit/Assertion/")

  add_hpx_unit_test("modules.assertion" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.modules.assertion.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.modules.assertion
                              tests.unit.modules.assertion.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.modules.assertion.${test}
                              ${test}_test)
endforeach()

set_tests_properties(tests.unit.modules.assertion.assert_fail
  PROPERTIES WILL_FAIL $<$<CONFIG:Debug>:true>)
