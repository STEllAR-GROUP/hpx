# Copyright (c) 2007-2012 Hartmut Kaiser
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
    ${HPX_COMPONENTS} memory
    CACHE INTERNAL "list of HPX components"
)

if(NOT HPX_WITH_DEFAULT_TARGETS)
  set(_exclude_from_all_flag EXCLUDE_FROM_ALL)
endif()

set(memory_headers hpx/components/performance_counters/memory/mem_counter.hpp)

set(memory_sources mem_counter_linux.cpp mem_counter_macosx.cpp
                   mem_counter_windows.cpp memory.cpp
)

add_hpx_component(
  memory INTERNAL_FLAGS
  FOLDER "Core/Components/Counters"
  INSTALL_HEADERS PLUGIN PREPEND_HEADER_ROOT
  HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
  HEADERS ${memory_headers}
  PREPEND_SOURCE_ROOT
  SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SOURCES ${memory_sources} ${_exclude_from_all_flag}
          ${HPX_WITH_UNITY_BUILD_OPTION}
)

add_hpx_pseudo_dependencies(
  components.performance_counters.memory memory_component
)

add_subdirectory(tests)
add_subdirectory(examples)
