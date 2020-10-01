# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(futures_headers
    hpx/futures/future.hpp
    hpx/futures/future_fwd.hpp
    hpx/futures/futures_factory.hpp
    hpx/futures/detail/future_data.hpp
    hpx/futures/detail/future_transforms.hpp
    hpx/futures/packaged_continuation.hpp
    hpx/futures/promise.hpp
    hpx/futures/traits/acquire_future.hpp
    hpx/futures/traits/acquire_shared_state.hpp
    hpx/futures/traits/detail/future_await_traits.hpp
    hpx/futures/traits/detail/future_traits.hpp
    hpx/futures/traits/future_access.hpp
    hpx/futures/traits/future_then_result.hpp
    hpx/futures/traits/future_traits.hpp
    hpx/futures/traits/get_remote_result.hpp
    hpx/futures/traits/is_future.hpp
    hpx/futures/traits/is_future_range.hpp
    hpx/futures/traits/is_future_tuple.hpp
    hpx/futures/traits/promise_local_result.hpp
    hpx/futures/traits/promise_remote_result.hpp
)

# cmake-format: off
set(futures_compat_headers
    hpx/lcos/future.hpp => hpx/future.hpp
    hpx/lcos/local/futures_factory.hpp => hpx/future.hpp
    hpx/lcos/local/packaged_continuation.hpp => hpx/future.hpp
    hpx/lcos/local/promise.hpp => hpx/future.hpp
    hpx/lcos_local/promise.hpp => hpx/future.hpp
    hpx/traits/acquire_future.hpp => hpx/future.hpp
    hpx/traits/acquire_shared_state.hpp => hpx/future.hpp
    hpx/traits/future_access.hpp => hpx/future.hpp
    hpx/traits/future_then_result.hpp => hpx/future.hpp
    hpx/traits/future_traits.hpp => hpx/future.hpp
    hpx/traits/get_remote_result.hpp => hpx/future.hpp
    hpx/traits/is_future.hpp => hpx/future.hpp
    hpx/traits/is_future_range.hpp => hpx/future.hpp
    hpx/traits/is_future_tuple.hpp => hpx/future.hpp
    hpx/traits/promise_local_result.hpp => hpx/future.hpp
    hpx/traits/promise_remote_result.hpp => hpx/future.hpp
)
# cmake-format: on

set(futures_sources future_data.cpp)

include(HPX_AddModule)
add_hpx_module(
  parallelism futures
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${futures_sources}
  HEADERS ${futures_headers}
  COMPAT_HEADERS ${futures_compat_headers}
  EXCLUDE_FROM_GLOBAL_HEADER "hpx/futures/detail/future_data.hpp"
                             "hpx/futures/detail/future_transforms.hpp"
  DEPENDENCIES hpx_core
  MODULE_DEPENDENCIES hpx_async_base
  CMAKE_SUBDIRS examples tests
)
