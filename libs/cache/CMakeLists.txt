# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(module_name cache)
string(TOUPPER ${module_name} upper_module_name)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(HPX.${module_name} CXX)

# Default location is $HPX_ROOT/libs/${module_name}/include
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

# Default location is $HPX_ROOT/libs/${module_name}/include_compatibility
set(${module_name}_compat_headers
  hpx/util/${module_name}/local_cache.hpp
  hpx/util/${module_name}/lru_cache.hpp
  hpx/util/${module_name}/entries/entry.hpp
  hpx/util/${module_name}/entries/fifo_entry.hpp
  hpx/util/${module_name}/entries/lfu_entry.hpp
  hpx/util/${module_name}/entries/lru_entry.hpp
  hpx/util/${module_name}/entries/size_entry.hpp
  hpx/util/${module_name}/policies/always.hpp
  hpx/util/${module_name}/statistics/local_full_statistics.hpp
  hpx/util/${module_name}/statistics/local_statistics.hpp
  hpx/util/${module_name}/statistics/no_statistics.hpp
)

# Default location is $HPX_ROOT/libs/${module_name}/src
set(${module_name}_sources
  cache.cpp
)

include(HPX_AddModule)
add_hpx_module(${module_name}
    DEPRECATION_WARNINGS
    COMPATIBILITY_HEADERS ON
    SOURCES ${${module_name}_sources}
    HEADERS ${${module_name}_headers}
    COMPAT_HEADERS ${${module_name}_compat_headers}
    DEPENDENCIES hpx_config
    CMAKE_SUBDIRS examples tests
)
