# Copyright (c) 2019-2021 The STE||AR-Group
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
    hpx/agas/agas_fwd.hpp
    hpx/agas/component_namespace.hpp
    hpx/agas/detail/bootstrap_component_namespace.hpp
    hpx/agas/detail/bootstrap_locality_namespace.hpp
    hpx/agas/detail/hosted_component_namespace.hpp
    hpx/agas/detail/hosted_locality_namespace.hpp
    hpx/agas/gva.hpp
    hpx/agas/locality_namespace.hpp
    hpx/agas/primary_namespace.hpp
    hpx/agas/server/component_namespace.hpp
    hpx/agas/server/locality_namespace.hpp
    hpx/agas/server/primary_namespace.hpp
    hpx/agas/server/symbol_namespace.hpp
    hpx/agas/symbol_namespace.hpp
)

# cmake-format: off
set(agas_compat_headers
    hpx/runtime/agas_fwd.hpp => hpx/agas/agas_fwd.hpp
    hpx/runtime/agas/gva.hpp => hpx/agas/gva.hpp
    hpx/runtime/agas/component_namespace.hpp => hpx/agas/component_namespace.hpp
    hpx/runtime/agas/locality_namespace.hpp => hpx/agas/locality_namespace.hpp
    hpx/runtime/agas/primary_namespace.hpp => hpx/agas/primary_namespace.hpp
    hpx/runtime/agas/detail/bootstrap_component_namespace.hpp => hpx/agas/detail/bootstrap_component_namespace.hpp
    hpx/runtime/agas/detail/bootstrap_locality_namespace.hpp => hpx/agas/detail/bootstrap_locality_namespace.hpp
    hpx/runtime/agas/detail/hosted_component_namespace.hpp => hpx/agas/detail/hosted_component_namespace.hpp
    hpx/runtime/agas/detail/hosted_locality_namespace.hpp => hpx/agas/detail/hosted_locality_namespace.hpp
    hpx/runtime/agas/server/component_namespace.hpp => hpx/agas/server/component_namespace.hpp
    hpx/runtime/agas/server/locality_namespace.hpp => hpx/agas/server/locality_namespace.hpp
    hpx/runtime/agas/server/primary_namespace.hpp => hpx/agas/server/primary_namespace.hpp
    hpx/runtime/agas/server/symbol_namespace.hpp => hpx/agas/server/symbol_namespace.hpp
    hpx/runtime/agas/symbol_namespace.hpp => hpx/agas/symbol_namespace.hpp
)
# cmake-format: on

set(agas_sources
    component_namespace.cpp
    detail/bootstrap_component_namespace.cpp
    detail/bootstrap_locality_namespace.cpp
    detail/hosted_component_namespace.cpp
    detail/hosted_locality_namespace.cpp
    locality_namespace.cpp
    primary_namespace.cpp
    server/component_namespace_server.cpp
    server/locality_namespace_server.cpp
    server/primary_namespace_server.cpp
    server/symbol_namespace_server.cpp
    symbol_namespace.cpp
)

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
  MODULE_DEPENDENCIES hpx_async_distributed hpx_collectives hpx_actions
                      hpx_components_base hpx_naming_base
  CMAKE_SUBDIRS examples tests
)
