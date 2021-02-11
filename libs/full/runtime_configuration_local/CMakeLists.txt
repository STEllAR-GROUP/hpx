# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_configuration_local_headers
    hpx/runtime_configuration_local/init_ini_data_local.hpp
    hpx/runtime_configuration_local/runtime_configuration_local.hpp
)

set(runtime_configuration_local_compat_headers)

set(runtime_configuration_local_sources init_ini_data_local.cpp
                                        runtime_configuration_local.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full runtime_configuration_local
  GLOBAL_HEADER_GEN ON
  HEADERS ${runtime_configuration_local_headers}
  SOURCES ${runtime_configuration_local_sources}
  DEPENDENCIES hpx_core
  MODULE_DEPENDENCIES hpx_ini
  CMAKE_SUBDIRS examples tests
)
