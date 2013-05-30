# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    apply_local
    apply_remote
    async_continue
    async_local
    async_remote
    barrier
    dataflow
    deferred_packaged_task
    for_each
    future
    future_wait
    local_barrier
    local_dataflow
    local_event
    local_mutex
    packaged_action
    promise
   )

set(async_continue_PARAMETERS LOCALITIES 2)

set(barrier_PARAMETERS THREADS_PER_LOCALITY 4)

set(dataflow_FLAGS DEPENDENCIES dataflow_component)
set(dataflow_PARAMETERS THREADS_PER_LOCALITY 4)

set(deferred_packaged_task_PARAMETERS THREADS_PER_LOCALITY 4)

set(future_PARAMETERS THREADS_PER_LOCALITY 4)

set(future_wait_PARAMETERS THREADS_PER_LOCALITY 4)

set(local_barrier_PARAMETERS THREADS_PER_LOCALITY 4)

set(local_event_PARAMETERS THREADS_PER_LOCALITY 4)

set(local_mutex_PARAMETERS THREADS_PER_LOCALITY 4)

set(packaged_action_PARAMETERS THREADS_PER_LOCALITY 4)

set(promise_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Unit/LCOs")

  add_hpx_unit_test(${test} ${${test}_PARAMETERS})  

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.lcos.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.lcos
                              tests.unit.lcos.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.lcos.${test}
                              ${test}_test_exe)
endforeach()

set_property(TARGET dataflow_test_exe APPEND
    PROPERTY COMPILE_DEFINITIONS
    "HPX_ACTION_ARGUMENT_LIMIT=10"
    "HPX_FUNCTION_ARGUMENT_LIMIT=13")

