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

set(agas_headers
    hpx/agas/agas_fwd.hpp hpx/agas/gva.hpp hpx/agas/primary_namespace.hpp
    hpx/agas/server/primary_namespace.hpp
)

# cmake-format: off
set(agas_compat_headers
    hpx/runtime/agas_fwd.hpp => hpx/agas/agas_fwd.hpp
    hpx/runtime/agas/gva.hpp => hpx/agas/gva.hpp
    hpx/runtime/agas/primary_namespace.hpp => hpx/agas/primary_namespace.hpp
    hpx/runtime/agas/server/primary_namespace.hpp => hpx/agas/server/primary_namespace.hpp
)
# cmake-format: on

set(agas_sources primary_namespace.cpp server/primary_namespace_server.cpp)

include(HPX_AddModule)
add_hpx_module(
  full agas
  COMPATIBILITY_HEADERS ON # Added in 1.6.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${agas_sources}
  HEADERS ${agas_headers}
  COMPAT_HEADERS ${agas_compat_headers}
  DEPENDENCIES hpx_core
  MODULE_DEPENDENCIES hpx_async_distributed hpx_actions hpx_components_base
                      hpx_naming_base
  CMAKE_SUBDIRS examples tests
)
