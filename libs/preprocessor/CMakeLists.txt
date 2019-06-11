# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

set(module_name preprocessor)
string(TOUPPER ${module_name} upper_module_name)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HPX_Message)
hpx_info("  preprocessor")

project(HPX.${module_name} CXX)

set(source_root ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(header_root ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(compat_header_root ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility)

# Full path here to avoid a prepend loop
set(${module_name}_headers
  ${header_root}/hpx/preprocessor.hpp
  ${header_root}/hpx/${module_name}/cat.hpp
  ${header_root}/hpx/${module_name}/config.hpp
  ${header_root}/hpx/${module_name}/expand.hpp
  ${header_root}/hpx/${module_name}/nargs.hpp
  ${header_root}/hpx/${module_name}/stringize.hpp
  ${header_root}/hpx/${module_name}/strip_parens.hpp
)

if(HPX_${upper_module_name}_WITH_COMPATIBILITY_HEADERS)
  set(${module_name}_compat_headers
    ${compat_header_root}/hpx/util/detail/pp/cat.hpp
    ${compat_header_root}/hpx/util/detail/pp/config.hpp
    ${compat_header_root}/hpx/util/detail/pp/expand.hpp
    ${compat_header_root}/hpx/util/detail/pp/nargs.hpp
    ${compat_header_root}/hpx/util/detail/pp/stringize.hpp
    ${compat_header_root}/hpx/util/detail/pp/strip_parens.hpp
  )
endif()

set(${module_name}_sources
  ${source_root}/preprocessor.cpp
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
    CMAKE_SUBDIRS src tests
)
