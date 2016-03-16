# Copyright (c) 2014-2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    bulk_async
    created_executor
    executor_parameters
    minimal_async_executor
    minimal_sync_executor
    minimal_timed_async_executor
    minimal_timed_sync_executor
    parallel_executor
    parallel_fork_executor
    persistent_executor_parameters
    sequential_executor
    service_executors
    shared_parallel_executor
    this_thread_executors
    thread_pool_attached_executors
    thread_pool_executors
    thread_pool_os_executors
    timed_parallel_executor
    timed_this_thread_executors
    timed_thread_pool_executors
   )

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Unit/Parallel/Executors")

  add_hpx_unit_test("parallel" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.parallel.executors.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parallel.executors
                              tests.unit.parallel.executors.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.parallel.executors.${test}
                              ${test}_test_exe)
endforeach()
