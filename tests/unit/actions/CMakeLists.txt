# Copyright (c) 2011 Bryce Adelstein-Lelbach
# Copyright (c) 2015 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests return_future)
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Actions"
  )

  add_hpx_unit_test("actions" ${test} ${${test}_PARAMETERS})

endforeach()

if(HPX_WITH_COMPILE_ONLY_TESTS)
  # add compile time tests
  set(compile_tests succeed_compile_const_ref_argument)

  if(HPX_WITH_FAIL_COMPILE_TESTS)
    set(fail_compile_tests
        fail_compile_non_const_array_argument fail_compile_const_array_argument
        fail_compile_non_const_pointer_argument
        fail_compile_const_pointer_argument fail_compile_non_const_ref_argument
    )
    foreach(fail_compile_test ${fail_compile_tests})
      set(${fail_compile_test}_FLAGS FAILURE_EXPECTED)
    endforeach()

    set(compile_tests ${compile_tests} ${fail_compile_tests})
  endif()

  foreach(compile_test ${compile_tests})
    set(sources ${compile_test}.cpp)

    source_group("Source Files" FILES ${sources})

    add_hpx_unit_compile_test(
      "actions" ${compile_test}
      SOURCES ${sources} ${${compile_test}_FLAGS}
      FOLDER "Tests/Unit/Actions/CompileOnly"
    )

  endforeach()
endif()
