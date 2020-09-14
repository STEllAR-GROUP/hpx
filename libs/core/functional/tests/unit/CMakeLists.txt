# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Function tests
set(function_tests
    # allocator_test
    bind_const_test
    bind_cv_test
    bind_dm2_test
    bind_dm3_test
    bind_dm_test
    bind_placeholder_test
    bind_rv_sp_test
    bind_rvalue_test
    bind_stateful_test
    bind_test
    contains_test
    function_args
    function_arith
    function_bind_test
    function_ref
    function_ref_wrapper
    function_target
    function_test
    mem_fn_derived_test
    mem_fn_dm_test
    # mem_fn_eq_test
    mem_fn_rv_test
    mem_fn_test
    mem_fn_unary_addr_test
    mem_fn_void_test
    nothrow_swap
    protect_test
    stateless_test
    sum_avg
    tag_invoke
)

foreach(test ${function_tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    NOLIBS
    DEPENDENCIES hpx_functional hpx_testing
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Core/Functional"
  )

  add_hpx_unit_test("modules.functional" ${test})

endforeach()
