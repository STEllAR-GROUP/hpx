# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(module_name config)
string(TOUPPER ${module_name} upper_module_name)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(HPX.${module_name} CXX)

# Default location is $HPX_ROOT/libs/${module_name}/include
set(${module_name}_headers
  hpx/config.hpp
  hpx/${module_name}/asio.hpp
  hpx/${module_name}/attributes.hpp
  hpx/${module_name}/autolink.hpp
  hpx/${module_name}/branch_hints.hpp
  hpx/${module_name}/compiler_fence.hpp
  hpx/${module_name}/compiler_native_tls.hpp
  hpx/${module_name}/compiler_specific.hpp
  hpx/${module_name}/constexpr.hpp
  hpx/${module_name}/debug.hpp
  hpx/${module_name}/emulate_deleted.hpp
  hpx/${module_name}/export_definitions.hpp
  hpx/${module_name}/forceinline.hpp
  hpx/${module_name}/lambda_capture.hpp
  hpx/${module_name}/manual_profiling.hpp
  hpx/${module_name}/threads_stack.hpp
  hpx/${module_name}/warnings_prefix.hpp
  hpx/${module_name}/warnings_suffix.hpp
  hpx/${module_name}/weak_symbol.hpp
)

# Default location is $HPX_ROOT/libs/${module_name}/src
set(${module_name}_sources
  version.cpp
)

include(HPX_AddModule)
add_hpx_module(${module_name}
    COMPATIBILITY_HEADERS OFF
    SOURCES ${${module_name}_sources}
    HEADERS ${${module_name}_headers}
    DEPENDENCIES hpx_preprocessor
    CMAKE_SUBDIRS examples tests
)
