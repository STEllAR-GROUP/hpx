# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Function tests
set(function_tests
    #allocator_test
  contains_test
  function_args
  function_arith
  function_bind_test
  function_ref
  function_ref_wrapper
  function_target
  function_test
  nothrow_swap
  stateless_test
  sum_avg
)

add_hpx_pseudo_target(tests.unit.modules.functional.function)

foreach(test ${function_tests})
  set(sources function/${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(${test}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    NOLIBS
    DEPENDENCIES hpx_functional hpx_testing
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Functional/Function")

  add_hpx_unit_test("modules.functional.function" ${test})

endforeach()
add_hpx_pseudo_dependencies(tests.unit.modules.functional tests.unit.modules.functional.function)
