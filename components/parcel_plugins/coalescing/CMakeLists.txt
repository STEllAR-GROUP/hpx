# Copyright (c) 2019-2022 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_PARCEL_COALESCING)
  return()
endif()

set(parcel_coalescing_headers
    hpx/include/parcel_coalescing.hpp hpx/parcel_coalescing/message_handler.hpp
    hpx/parcel_coalescing/counter_registry.hpp
    hpx/parcel_coalescing/message_buffer.hpp
)

set(parcel_coalescing_sources
    coalescing_message_handler.cpp coalescing_counter_registry.cpp
    performance_counters.cpp
)

if(TARGET APEX::apex)
  set(additional_dependencies APEX::apex)
endif()

add_hpx_library(
  parcel_coalescing INTERNAL_FLAGS
  FOLDER "Core/Plugins/MessageHandlers"
  INSTALL_HEADERS PLUGIN PREPEND_HEADER_ROOT PREPEND_SOURCE_ROOT
  ${HPX_WITH_UNITY_BUILD_OPTION} ${HPX_WITH_CXX_MODULES_OPTION}
  HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
  HEADERS ${parcel_coalescing_headers}
  SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SOURCES ${parcel_coalescing_sources}
  DEPENDENCIES ${additional_dependencies}
)

target_compile_definitions(
  parcel_coalescing PRIVATE HPX_PARCEL_COALESCING_MODULE_EXPORTS
)

add_hpx_pseudo_dependencies(
  components.parcel_plugins.coalescing parcel_coalescing
)

add_subdirectory(tests)
