# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(include_local_headers
    hpx/local/algorithm.hpp
    hpx/local/any.hpp
    hpx/local/barrier.hpp
    hpx/local/channel.hpp
    hpx/local/chrono.hpp
    hpx/local/condition_variable.hpp
    hpx/local/exception.hpp
    hpx/local/execution.hpp
    hpx/local/functional.hpp
    hpx/local/future.hpp
    hpx/local/latch.hpp
    hpx/local/memory.hpp
    hpx/local/mutex.hpp
    hpx/local/numeric.hpp
    hpx/local/optional.hpp
    hpx/local/runtime.hpp
    hpx/local/semaphore.hpp
    hpx/local/shared_mutex.hpp
    hpx/local/stop_token.hpp
    hpx/local/system_error.hpp
    hpx/local/task_block.hpp
    hpx/local/thread.hpp
    hpx/local/tuple.hpp
    hpx/local/type_traits.hpp
    hpx/local/unwrap.hpp
)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local include_local
  GLOBAL_HEADER_GEN OFF
  HEADERS ${include_local_headers}
  MODULE_DEPENDENCIES
    hpx_algorithms
    hpx_async_base
    hpx_async_combinators
    hpx_async_local
    hpx_errors
    hpx_execution
    hpx_executors
    hpx_futures
    hpx_lcos_local
    hpx_runtime_local
  CMAKE_SUBDIRS examples tests
)
