# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(distributed_executors_headers
    hpx/distributed_executors/distribution_policy_executor.hpp
)

set(distributed_executors_compat_headers
    hpx/parallel/executors/distribution_policy_executor.hpp
)

include(HPX_AddModule)
add_hpx_module(
  distributed_executors
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  HEADERS ${distributed_executors_headers}
  COMPAT_HEADERS ${distributed_executors_compat_headers}
  DEPENDENCIES hpx_async_distributed hpx_config hpx_execution hpx_functional
               hpx_type_support
  CMAKE_SUBDIRS examples tests
)
