# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    exception
    handled_exception
   )

if(HPX_WITH_FAIL_COMPILE_TESTS)
  set(fail_compile_tests
      unhandled_exception
     )
  foreach(fail_compile_test ${fail_compile_tests})
    set(${fail_compile_test}_PARAMETERS FAILURE_EXPECTED)
  endforeach()

  set(tests ${tests} ${fail_compile_tests})
endif()

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
                     FOLDER "Tests/Unit/Diagnostics")

  add_hpx_unit_test("diagnostics" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.diagnostics.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.diagnostics
                              tests.unit.diagnostics.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.diagnostics.${test}
                              ${test}_test)
endforeach()

