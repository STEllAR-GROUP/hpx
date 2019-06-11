# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(module_name cache)
string(TOUPPER ${module_name} upper_module_name)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HPX_Message)

hpx_info("  cache")

project(HPX.${module_name} CXX)

set(source_root ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(header_root ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(compat_header_root ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility)

# No full path here cause easier to create a global include file inside
# add_hpx_module and because cmake 3.12 not always used
set(${module_name}_headers
  hpx/${module_name}/local_cache.hpp
  hpx/${module_name}/lru_cache.hpp
  hpx/${module_name}/entries/entry.hpp
  hpx/${module_name}/entries/fifo_entry.hpp
  hpx/${module_name}/entries/lfu_entry.hpp
  hpx/${module_name}/entries/lru_entry.hpp
  hpx/${module_name}/entries/size_entry.hpp
  hpx/${module_name}/policies/always.hpp
  hpx/${module_name}/statistics/local_full_statistics.hpp
  hpx/${module_name}/statistics/local_statistics.hpp
  hpx/${module_name}/statistics/no_statistics.hpp
)

if(HPX_${upper_module_name}_WITH_COMPATIBILITY_HEADERS)
  set(${module_name}_compat_headers
    ${compat_header_root}/hpx/util/${module_name}/local_cache.hpp
    ${compat_header_root}/hpx/util/${module_name}/lru_cache.hpp
    ${compat_header_root}/hpx/util/${module_name}/entries/entry.hpp
    ${compat_header_root}/hpx/util/${module_name}/entries/fifo_entry.hpp
    ${compat_header_root}/hpx/util/${module_name}/entries/lfu_entry.hpp
    ${compat_header_root}/hpx/util/${module_name}/entries/lru_entry.hpp
    ${compat_header_root}/hpx/util/${module_name}/entries/size_entry.hpp
    ${compat_header_root}/hpx/util/${module_name}/policies/always.hpp
    ${compat_header_root}/hpx/util/${module_name}/statistics/local_full_statistics.hpp
    ${compat_header_root}/hpx/util/${module_name}/statistics/local_statistics.hpp
    ${compat_header_root}/hpx/util/${module_name}/statistics/no_statistics.hpp
  )
endif()

set(${module_name}_sources
  ${source_root}/cache.cpp
)

include(HPX_AddModule)
add_hpx_module(${module_name}
    DEPRECATION_WARNINGS
    COMPATIBILITY_HEADERS
    FOLDER ${CMAKE_CURRENT_SOURCE_DIR}
    SOURCE_ROOT ${source_root}
    HEADER_ROOT ${header_root}
    COMPAT_HEADER_ROOT ${compat_header_root}
    SOURCES ${${module_name}_sources}
    HEADERS ${${module_name}_headers}
    COMPAT_HEADERS ${${module_name}_compat_headers}
    DEPENDENCIES hpx_config
    CMAKE_SUBDIRS examples tests
)
