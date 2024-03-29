# Copyright (c) 2015-2023 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

set(HPX_COMPONENTS
    ${HPX_COMPONENTS} unordered
    CACHE INTERNAL "list of HPX components"
)

set(unordered_headers
    hpx/components/containers/unordered/partition_unordered_map_component.hpp
    hpx/components/containers/unordered/unordered_map.hpp
    hpx/components/containers/unordered/unordered_map_segmented_iterator.hpp
    hpx/include/unordered_map.hpp
)

set(unordered_sources partition_unordered_map_component.cpp)

add_hpx_component(
  unordered INTERNAL_FLAGS
  FOLDER "Core/Components/Containers"
  PREPEND_HEADER_ROOT INSTALL_HEADERS
  INSTALL_COMPONENT runtime
  HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
  HEADERS ${unordered_headers}
  PREPEND_SOURCE_ROOT
  SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SOURCES ${unordered_sources} ${HPX_WITH_UNITY_BUILD_OPTION}
)

add_hpx_pseudo_dependencies(components.containers.unordered unordered_component)

add_subdirectory(tests)
add_subdirectory(examples)
