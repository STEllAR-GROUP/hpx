# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    async_callback_non_deduced_context
    future_hang_on_get
    dataflow_launch
    regression_future_790
    regression_future_unwrap_878
    regression_set_hpx_limit_771
    regression_set_hpx_limit_798
   )

# MSVC 2010 is not able to cope with this test
if(NOT MSVC10)
  set(dataflow_future_swap_FLAGS DEPENDENCIES iostreams_component)
  set(dataflow_future_swap2_FLAGS DEPENDENCIES iostreams_component)
  set(dataflow_future_swap_PARAMETERS THREADS_PER_LOCALITY 4)
  set(tests ${tests}
      dataflow_future_swap
      regression_dataflow_791
     )
  if(HPX_HAVE_CXX11_LAMBDAS)
    set(tests ${tests}
        dataflow_const_functor
        dataflow_future_swap2
        dataflow_using
       )
  endif()
endif()

set(async_callback_non_deduced_context_PARAMETERS THREADS_PER_LOCALITY 4)
set(future_hang_on_get_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)
set(dataflow_launch_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)

set(regression_set_hpx_limit_771_FLAGS DEPENDENCIES dataflow_component)

if(HPX_HAVE_CXX11_LAMBDAS)
  set(future_hang_on_wait_with_callback_PARAMETERS THREADS_PER_LOCALITY 4)
  set(tests
    ${tests}
    future_hang_on_wait_with_callback)
endif()

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Regressions/LCOs")

  add_hpx_regression_test("lcos" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.lcos.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.lcos
                              tests.regressions.lcos.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.lcos.${test}
                              ${test}_test_exe)
endforeach()

