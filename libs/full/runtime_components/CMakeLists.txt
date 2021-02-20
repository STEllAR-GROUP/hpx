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

set(runtime_components_headers
    hpx/runtime_components/component_factory.hpp
    hpx/runtime_components/component_registry.hpp
    hpx/runtime_components/copy_component.hpp
    hpx/runtime_components/create_component_helpers.hpp
    hpx/runtime_components/derived_component_factory.hpp
    hpx/runtime_components/migrate_component.hpp
    hpx/runtime_components/new.hpp
    hpx/runtime_components/server/copy_component.hpp
    hpx/runtime_components/server/migrate_component.hpp
)

# cmake-format: off
set(runtime_components_compat_headers
    hpx/runtime/components/component_factory.hpp => hpx/modules/runtime_components.hpp
    hpx/runtime/components/component_registry.hpp => hpx/modules/runtime_components.hpp
    hpx/runtime/components/copy_component.hpp => hpx/modules/runtime_components.hpp
    hpx/runtime/components/create_component_helpers.hpp => hpx/modules/runtime_components.hpp
    hpx/runtime/components/derived_component_factory.hpp => hpx/modules/runtime_components.hpp
    hpx/runtime/components/migrate_component.hpp => hpx/modules/runtime_components.hpp
    hpx/runtime/components/new.hpp => hpx/modules/runtime_components.hpp
    hpx/runtime/components/server/copy_component.hpp => hpx/modules/runtime_components.hpp
    hpx/runtime/components/server/migrate_component.hpp => hpx/modules/runtime_components.hpp
)
# cmake-format: on

set(runtime_components_sources component_registry.cpp)

include(HPX_AddModule)
add_hpx_module(
  full runtime_components
  COMPATIBILITY_HEADERS ON # Added in 1.7.0
  GLOBAL_HEADER_GEN ON
  HEADERS ${runtime_components_headers}
  SOURCES ${runtime_components_sources}
  COMPAT_HEADERS ${runtime_components_compat_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  MODULE_DEPENDENCIES hpx_async_colocated hpx_components
  CMAKE_SUBDIRS examples tests
)
