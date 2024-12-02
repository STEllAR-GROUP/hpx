# Copyright (c) 2007-2023 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

# The memory counters are not supported on FreeBSD
if(${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
  return()
endif()

set(HPX_COMPONENTS
    ${HPX_COMPONENTS} memory_counters
    CACHE INTERNAL "list of HPX components"
)

set(memory_headers
    hpx/components/performance_counters/memory_counters/mem_counter.hpp
)

set(memory_sources mem_counter_linux.cpp mem_counter_macosx.cpp
                   mem_counter_windows.cpp memory.cpp
)

add_hpx_component(
  memory_counters INTERNAL_FLAGS
  FOLDER "Core/Components/Counters"
  INSTALL_HEADERS PLUGIN PREPEND_HEADER_ROOT
  INSTALL_COMPONENT runtime
  HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
  HEADERS ${memory_headers}
  PREPEND_SOURCE_ROOT
  SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SOURCES ${memory_sources} ${HPX_WITH_UNITY_BUILD_OPTION}
)

# see #6579
if(MSVC AND Boost_VERSION_STRING VERSION_LESS "1.79.0")
  target_compile_definitions(memory_counters_component PRIVATE NOMINMAX)
endif()

add_hpx_pseudo_dependencies(
  components.performance_counters.memory_counters memory_counters_component
)

add_subdirectory(tests)
add_subdirectory(examples)
