# Copyright (c) 2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    function_bind_test
    contains_test
    function_arith
    function_args
    function_ref
    function_target
    function_test
    nothrow_swap
    stateless_test
    sum_avg
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
                     FOLDER "Tests/Unit/Util/Function")

  add_hpx_unit_test("util.function" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.util.functiondir.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util.functiondir
                              tests.unit.util.functiondir.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util.functiondir.${test}
                              ${test}_test_exe)
endforeach()


