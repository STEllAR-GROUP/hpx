# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

# Default location is $HPX_ROOT/libs/naming_base/include
set(naming_base_headers hpx/naming_base/gid_type.hpp
                        hpx/naming_base/naming_base.hpp
)

set(naming_base_compat_headers)

set(naming_base_sources gid_type.cpp)

include(HPX_AddModule)
add_hpx_module(
  full naming_base
  COMPATIBILITY_HEADERS OFF
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  HEADERS ${naming_base_headers}
  SOURCES ${naming_base_sources}
  DEPENDENCIES hpx_core
  CMAKE_SUBDIRS examples tests
)
