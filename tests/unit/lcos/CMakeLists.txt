# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    apply_colocated
    apply_local
    apply_local_executor
    apply_remote
    apply_remote_client
    async_cb_colocated
    async_cb_remote
    async_cb_remote_client
    async_colocated
    async_continue
    async_continue_cb
    async_continue_cb_colocated
    async_continue_colocated
    async_local
    async_local_executor
    async_remote
    async_remote_client
    async_unwrap_result
    channel
    client_then
    future
    future_ref
    future_then
    future_then_executor
    future_wait
    local_promise_allocator
    make_future
    make_ready_future
    packaged_action
    promise
    promise_allocator
    promise_emplace
    remote_dataflow
    shared_future
    split_shared_future
    sync_remote
    use_allocator
    wait_all_std_array
    wait_any_std_array
    when_all
    when_all_std_array
    when_any
    when_any_std_array
    when_each
    when_some
    when_some_std_array
   )

if(HPX_WITH_AWAIT)
  set(tests ${tests} await)
  set(await_PARAMETERS THREADS_PER_LOCALITY 4)
endif()

set(apply_colocated_PARAMETERS LOCALITIES 2)
set(apply_remote_PARAMETERS LOCALITIES 2)
set(apply_remote_client_PARAMETERS LOCALITIES 2)
set(apply_local_PARAMETERS THREADS_PER_LOCALITY 4)
set(apply_local_executor_PARAMETERS THREADS_PER_LOCALITY 4)
set(async_cb_colocated_PARAMETERS LOCALITIES 2)
set(async_local_PARAMETERS THREADS_PER_LOCALITY 4)
set(async_local_executor_PARAMETERS THREADS_PER_LOCALITY 4)

set(async_continue_PARAMETERS LOCALITIES 2)
set(async_continue_cb_PARAMETERS LOCALITIES 2)
set(async_continue_cb_colocated_PARAMETERS LOCALITIES 2)
set(async_remote_PARAMETERS LOCALITIES 2)
set(async_remote_client_PARAMETERS LOCALITIES 2)
set(async_cb_remote_PARAMETERS LOCALITIES 2)
set(async_cb_remote_client_PARAMETERS LOCALITIES 2)

set(future_PARAMETERS THREADS_PER_LOCALITY 4)
set(future_then_PARAMETERS THREADS_PER_LOCALITY 4)
set(future_then_executor_PARAMETERS THREADS_PER_LOCALITY 4)
set(future_wait_PARAMETERS THREADS_PER_LOCALITY 4)

set(remote_dataflow_PARAMETERS THREADS_PER_LOCALITY 4)
set(remote_dataflow_PARAMETERS LOCALITIES 2)

set(packaged_action_PARAMETERS THREADS_PER_LOCALITY 4)

set(promise_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/LCOs")

  # add example executable
  add_hpx_executable(${test}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name})

  add_hpx_unit_test("lcos" ${test} ${${test}_PARAMETERS})

endforeach()
