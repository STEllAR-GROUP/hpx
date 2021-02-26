# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_components_base_headers
    hpx/runtime_components_base/component_factory.hpp
    hpx/runtime_components_base/component_registry.hpp
    hpx/runtime_components_base/derived_component_factory.hpp
)

# cmake-format: off
set(runtime_components_base_compat_headers
    hpx/runtime/components/component_factory.hpp => hpx/modules/runtime_components_base.hpp
    hpx/runtime/components/component_registry.hpp => hpx/modules/runtime_components_base.hpp
    hpx/runtime/components/derived_component_factory.hpp => hpx/modules/runtime_components_base.hpp
)
# cmake-format: on

set(runtime_components_base_sources component_registry.cpp)

include(HPX_AddModule)
add_hpx_module(
  full runtime_components_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${runtime_components_base_sources}
  HEADERS ${runtime_components_base_headers}
  COMPAT_HEADERS ${runtime_components_base_compat_headers}
  DEPENDENCIES hpx_async_colocated
  CMAKE_SUBDIRS examples tests
)
