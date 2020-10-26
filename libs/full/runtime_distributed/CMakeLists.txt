# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_distributed_headers
    hpx/runtime_distributed.hpp hpx/runtime_distributed/get_num_localities.hpp
    hpx/runtime_distributed/find_localities.hpp
)

# cmake-format: off
set(runtime_distributed_compat_headers
    hpx/runtime/get_num_localities.hpp => hpx/runtime_distributed.hpp
)
# cmake-format: on

set(runtime_distributed_sources)

include(HPX_AddModule)
add_hpx_module(
  full runtime_distributed
  COMPATIBILITY_HEADERS ON # added in hpx V1.6.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN OFF
  SOURCES ${runtime_distributed_sources}
  HEADERS ${runtime_distributed_headers}
  COMPAT_HEADERS ${runtime_distributed_compat_headers}
  DEPENDENCIES hpx_core
  MODULE_DEPENDENCIES hpx_naming hpx_naming_base hpx_performance_counters
                      hpx_runtime_local
  CMAKE_SUBDIRS examples tests
)
