# Copyright (c) 2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    bind_test
    contains_test
    function_arith
    function_ref
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
                     FOLDER "Tests/Unit/Util/Function")

  add_hpx_unit_test(${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.util.function.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util.function
                              tests.unit.util.function.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.util.function.${test}
                              ${test}_test_exe)
endforeach()


