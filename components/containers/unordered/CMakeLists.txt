# Copyright (c) 2015 Hartmut Kaiser
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

if(NOT HPX_WITH_DEFAULT_TARGETS)
  set(_exclude_from_all_flag EXCLUDE_FROM_ALL)
endif()

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
  INSTALL_HEADERS PREPEND_HEADER_ROOT
  HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
  HEADERS ${unordered_headers}
  PREPEND_SOURCE_ROOT
  SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SOURCES ${unordered_sources} ${_exclude_from_all_flag}
          ${HPX_WITH_UNITY_BUILD_OPTION}
)

add_hpx_pseudo_dependencies(components.containers.unordered unordered_component)

add_subdirectory(tests)
add_subdirectory(examples)
