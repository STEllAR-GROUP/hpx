# Copyright (c) 2007-2015 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    after_588
    async_action_1813
    async_callback_with_bound_callback
    async_callback_non_deduced_context
    async_unwrap_1037
    call_promise_get_gid_more_than_once
    dataflow_791
    dataflow_action_2008
    dataflow_future_swap
    dataflow_launch_775
    future_790
    future_hang_on_get_629
    future_hang_on_then_629
    future_timed_wait_1025
    future_serialization_1898
    future_unwrap_878
    future_unwrap_1182
    ignore_while_locked_1485
    lifetime_588
    lifetime_588_1
    multiple_gather_ops_2001
    promise_leak_996
    promise_1620
    receive_buffer_1733
    safely_destroy_promise_1481
    set_hpx_limit_798
    shared_mutex_1702
    shared_stated_leaked_1211
    wait_all_hang_1946
    wait_for_1751
    when_all_vectors_1623
   )

set(promise_1620_FLAGS DEPENDENCIES iostreams_component)
set(dataflow_future_swap_FLAGS DEPENDENCIES iostreams_component)
set(dataflow_future_swap_PARAMETERS THREADS_PER_LOCALITY 4)
set(after_588_PARAMETERS LOCALITIES 2)
set(async_action_1813_PARAMETERS LOCALITIES 2)
set(async_callback_with_bound_callback_PARAMETERS LOCALITIES 2)
set(async_callback_non_deduced_context_PARAMETERS THREADS_PER_LOCALITY 4)
set(future_hang_on_get_629_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)
set(dataflow_launch_775_PARAMETERS THREADS_PER_LOCALITY 2)
set(ignore_while_locked_1485_PARAMETERS THREADS_PER_LOCALITY 2)
set(lifetime_588_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)
set(lifetime_588_1_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)
set(multiple_gather_ops_2001_PARAMETERS LOCALITIES 2)
set(receive_buffer_1733_PARAMETERS LOCALITIES 2)
set(safely_destroy_cv_1481_PARAMETERS THREADS_PER_LOCALITY 2)
set(shared_mutex_1702_PARAMETERS THREADS_PER_LOCALITY 4)
set(wait_for_1751_PARAMETERS THREADS_PER_LOCALITY 4)
set(wait_all_hang_1946_PARAMETERS THREADS_PER_LOCALITY 8)
set(wait_all_hang_1946_FLAGS DEPENDENCIES iostreams_component)

if(HPX_WITH_CXX11_LAMBDAS)
  set(tests ${tests}
      dataflow_const_functor_773
      dataflow_future_swap2
      dataflow_recursion_1613
      dataflow_using_774
      exception_from_continuation_1613
      future_hang_on_wait_with_callback_629
     )
  set(dataflow_future_swap2_FLAGS DEPENDENCIES iostreams_component)
  set(dataflow_recursion_1613_PARAMETERS THREADS_PER_LOCALITY 2)
  set(exception_from_continuation_1613_PARAMETERS THREADS_PER_LOCALITY 2)
  set(future_hang_on_wait_with_callback_629_PARAMETERS THREADS_PER_LOCALITY 4)
endif()

# Create test cases
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
                     FOLDER "Tests/Regressions/LCOs")

  add_hpx_regression_test("lcos" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.lcos_dir.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.lcos_dir
                              tests.regressions.lcos_dir.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.lcos_dir.${test}
                              ${test}_test_exe)
endforeach()

add_hpx_test(
  "tests.regressions.lcos" after_588_4
  EXECUTABLE after_588
  LOCALITIES 2
  THREADS_PER_LOCALITY 4)

# compile only tests
set(compile_tests
    wait_all_std_array_2035
   )

foreach(compile_test ${compile_tests})
  set(sources
      ${compile_test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_unit_compile_test(
    "lcos_regressions"
    ${compile_test}
    SOURCES ${sources}
    ${${compile_test}_FLAGS}
    FOLDER "Tests/Regressions/LCOs/CompileOnly")

    # add a custom target for this example
    add_hpx_pseudo_target(tests.regressions.lcos_dir.${compile_test})

    # make pseudo-targets depend on master pseudo-target
    add_hpx_pseudo_dependencies(tests.regressions.lcos_dir
      tests.regressions.lcos_dir.${compile_test})

    # add dependencies to pseudo-target
    add_hpx_pseudo_dependencies(tests.regressions.lcos_dir.${compile_test}
      "tests.regressions.lcos_dir.${compile_test}")

endforeach()
