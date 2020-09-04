# Copyright (c) 2015 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

set(HPX_COMPONENTS
    ${HPX_COMPONENTS} component_storage
    CACHE INTERNAL "list of HPX components"
)

if(NOT HPX_WITH_DEFAULT_TARGETS)
  set(_exclude_from_all_flag EXCLUDE_FROM_ALL)
endif()

set(component_storage_headers
    hpx/components/component_storage/server/component_storage.hpp
    hpx/components/component_storage/server/migrate_from_storage.hpp
    hpx/components/component_storage/server/migrate_to_storage.hpp
    hpx/components/component_storage/component_storage.hpp
    hpx/components/component_storage/export_definitions.hpp
    hpx/components/component_storage/migrate_from_storage.hpp
    hpx/components/component_storage/migrate_to_storage.hpp
    hpx/include/component_storage.hpp
)

set(component_storage_sources server/component_storage_server.cpp
                              component_module.cpp component_storage.cpp
)

add_hpx_component(
  component_storage INTERNAL_FLAGS
  FOLDER "Core/Components/IO"
  INSTALL_HEADERS PREPEND_HEADER_ROOT
  HEADER_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include
  HEADERS ${component_storage_headers}
  PREPEND_SOURCE_ROOT
  SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src
  SOURCES ${component_storage_sources}
  DEPENDENCIES unordered_component ${_exclude_from_all_flag}
               ${HPX_WITH_UNITY_BUILD_OPTION}
)

target_compile_definitions(
  component_storage_component PRIVATE HPX_MIGRATE_TO_STORAGE_MODULE_EXPORTS
)

add_hpx_pseudo_dependencies(
  components.component_storage component_storage_component
)

add_subdirectory(tests)
add_subdirectory(examples)
