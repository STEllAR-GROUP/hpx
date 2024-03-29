# Copyright (c) 2007-2021 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(components)

set(tests
    async_action_1813
    async_callback_with_bound_callback
    async_callback_non_deduced_context
    async_deferred_1523
    async_unwrap_1037
    component_action_move_semantics
    dataflow_791
    dataflow_4436
    dataflow_action_2008
    dataflow_const_functor_773
    dataflow_future_swap2
    dataflow_future_swap
    dataflow_launch_775
    dataflow_recursion_1613
    dataflow_using_774
    make_continuation_1615
    missing_include_2958
    plain_action_1330
    plain_action_1550
    plain_action_move_semantics
    return_future_2847
    shared_future_serialization_1402
    future_hang_on_get_629
    future_hang_on_then_629
    future_hang_on_wait_with_callback_629
    future_serialization_1898
    future_timed_wait_1025
    return_non_default_constructible_2847
)

set(async_action_1813_PARAMETERS LOCALITIES 2)
set(async_callback_with_bound_callback_PARAMETERS LOCALITIES 2)
set(async_callback_non_deduced_context_PARAMETERS THREADS_PER_LOCALITY 4)
set(dataflow_future_swap_FLAGS DEPENDENCIES iostreams_component)
set(dataflow_future_swap_PARAMETERS THREADS_PER_LOCALITY 4)
set(dataflow_future_swap2_FLAGS DEPENDENCIES iostreams_component)
set(dataflow_launch_775_PARAMETERS THREADS_PER_LOCALITY 2)
set(dataflow_recursion_1613_PARAMETERS THREADS_PER_LOCALITY 2)
set(plain_action_1330_FLAGS DEPENDENCIES iostreams_component)
set(return_future_2847_PARAMETERS LOCALITIES 2)
set(future_hang_on_get_629_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)
set(future_hang_on_wait_with_callback_629_PARAMETERS THREADS_PER_LOCALITY 4)
set(component_action_move_semantics_FLAGS DEPENDENCIES
                                          action_move_semantics_component
)
set(component_action_move_semantics_PARAMETERS LOCALITIES 2
                                               THREADS_PER_LOCALITY 1
)
set(plain_action_move_semantics_FLAGS DEPENDENCIES
                                      action_move_semantics_component
)
set(plain_action_move_semantics_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 1)
set(return_non_default_constructible_2847_PARAMETERS LOCALITIES 2)

# Create test cases
foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Regressions/Modules/Full/AsyncDistributed"
  )

  add_hpx_regression_test(
    "modules.async_distributed" ${test} ${${test}_PARAMETERS}
  )
endforeach()

# compile only tests
if(HPX_WITH_COMPILE_ONLY_TESTS)
  set(compile_tests dataflow_future_ref_3639)

  foreach(compile_test ${compile_tests})
    set(sources ${compile_test}.cpp)

    source_group("Source Files" FILES ${sources})

    add_hpx_regression_compile_test(
      "modules.async_distributed" ${compile_test}
      SOURCES ${sources} ${${compile_test}_FLAGS}
      FOLDER "Tests/Regressions/Modules/Full/AsyncDistributed/CompileOnly"
    )
  endforeach()
endif()
