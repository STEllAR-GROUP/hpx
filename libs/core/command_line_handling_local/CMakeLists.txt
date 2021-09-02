# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(command_line_handling_local_headers
    hpx/command_line_handling_local/command_line_handling_local.hpp
    hpx/command_line_handling_local/late_command_line_handling_local.hpp
    hpx/command_line_handling_local/parse_command_line_local.hpp
)

set(command_line_handling_local_sources
    command_line_handling_local.cpp late_command_line_handling_local.cpp
    parse_command_line_local.cpp
)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local command_line_handling_local
  GLOBAL_HEADER_GEN ON
  SOURCES ${command_line_handling_local_sources}
  HEADERS ${command_line_handling_local_headers}
  DEPENDENCIES
  MODULE_DEPENDENCIES
    hpx_config_local
    hpx_program_options
    hpx_runtime_configuration
    hpx_asio
    hpx_debugging
    hpx_errors
    hpx_filesystem
    hpx_format
    hpx_topology
    hpx_util
    hpx_version_local
  CMAKE_SUBDIRS examples tests
)
