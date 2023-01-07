# Copyright (c) 2022 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT (HPX_WITH_SYCL))
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_sycl_headers
    hpx/async_sycl/detail/sycl_event_callback.hpp
    hpx/async_sycl/sycl_future.hpp hpx/async_sycl/sycl_executor.hpp
    hpx/async_sycl/sycl_polling_helper.hpp
)

# cmake-format: off
set(async_sycl_compat_headers)
# cmake-format: on

set(async_sycl_sources sycl_event_callback.cpp sycl_future.cpp)

hpx_info("Building HPX SYCL Module")

include(HPX_AddModule)
add_hpx_module(
  core async_sycl
  GLOBAL_HEADER_GEN ON
  SOURCES ${async_sycl_sources}
  HEADERS ${async_sycl_headers}
  COMPAT_HEADERS ${async_sycl_compat_headers}
  DEPENDENCIES
    hpx_allocator_support
    hpx_assertion
    hpx_concurrency
    hpx_config
    hpx_debugging
    hpx_errors
    hpx_execution_base
    hpx_futures
    hpx_memory
    hpx_runtime_local
    hpx_threading_base
  CMAKE_SUBDIRS examples tests
)
# make it compile with sycl
if(HPX_WITH_HIPSYCL)
  add_sycl_to_target(TARGET hpx_async_sycl SOURCES ${async_sycl_sources})
else()
  separate_arguments(HPX_WITH_SYCL_FLAGS UNIX_COMMAND "${HPX_WITH_SYCL_FLAGS}")
  target_compile_options(hpx_async_sycl PUBLIC -fsycl ${HPX_WITH_SYCL_FLAGS})
  target_link_options(hpx_async_sycl PUBLIC -fsycl ${HPX_WITH_SYCL_FLAGS})
endif()
