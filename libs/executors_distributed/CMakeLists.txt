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

set(executors_distributed_headers
    hpx/executors_distributed/distribution_policy_executor.hpp
)

set(executors_distributed_compat_headers
    hpx/parallel/executors/distribution_policy_executor.hpp
)

include(HPX_AddModule)
add_hpx_module(
  executors_distributed
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  HEADERS ${executors_distributed_headers}
  COMPAT_HEADERS ${executors_distributed_compat_headers}
  DEPENDENCIES hpx_async_distributed hpx_config hpx_execution hpx_functional
               hpx_type_support
  CMAKE_SUBDIRS examples tests
)
