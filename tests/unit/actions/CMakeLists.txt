# Copyright (c) 2011 Bryce Adelstein-Lelbach
# Copyright (c) 2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    return_future
   )

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
                     FOLDER "Tests/Unit/Actions")

  add_hpx_unit_test("actions" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.actions.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.actions
                              tests.unit.actions.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.actions.${test}
                              ${test}_test_exe)
endforeach()

if(HPX_WITH_COMPILE_ONLY_TESTS)
  # add compile time tests
  set(compile_tests
      fail_compile_non_const_array_argument
      fail_compile_const_array_argument
      fail_compile_non_const_pointer_argument
      fail_compile_const_pointer_argument
      fail_compile_non_const_ref_argument
      succeed_compile_const_ref_argument
     )

  set(fail_compile_non_const_array_argument_FLAGS FAILURE_EXPECTED)
  set(fail_compile_const_array_argument_FLAGS FAILURE_EXPECTED)
  set(fail_compile_non_const_ref_argument_FLAGS FAILURE_EXPECTED)
  set(fail_compile_non_const_pointer_argument_FLAGS FAILURE_EXPECTED)
  set(fail_compile_const_pointer_argument_FLAGS FAILURE_EXPECTED)

  foreach(compile_test ${compile_tests})
    set(sources
        ${compile_test}.cpp)

    source_group("Source Files" FILES ${sources})

    add_hpx_unit_compile_test(
      "actions"
      ${compile_test}
      SOURCES ${sources}
      ${${compile_test}_FLAGS}
      FOLDER "Tests/Unit/Actions/CompileOnly")

    # add a custom target for this example
    add_hpx_pseudo_target(tests.unit.actions.${compile_test})

    # make pseudo-targets depend on master pseudo-target
    add_hpx_pseudo_dependencies(tests.unit.actions
                                tests.unit.actions.${compile_test})

    # add dependencies to pseudo-target
    add_hpx_pseudo_dependencies(tests.unit.actions.${compile_test}
                                "tests.unit.actions.${compile_test}")

  endforeach()
endif()

