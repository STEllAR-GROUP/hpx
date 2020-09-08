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

set(lcos_distributed_headers
    hpx/distributed/channel.hpp hpx/lcos_distributed/channel.hpp
    hpx/lcos_distributed/server/channel.hpp
)

set(lcos_distributed_compat_headers hpx/lcos/channel.hpp
                                    hpx/lcos/server/channel.hpp
)

set(lcos_distributed_sources)

include(HPX_AddModule)
add_hpx_module(
  full lcos_distributed
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${lcos_distributed_sources}
  HEADERS ${lcos_distributed_headers}
  COMPAT_HEADERS ${lcos_distributed_compat_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  MODULE_DEPENDENCIES hpx_actions hpx_async_distributed
  CMAKE_SUBDIRS examples tests
)
