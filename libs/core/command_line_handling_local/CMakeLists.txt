# Copyright (c) 2019-2024 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Enable reading JSON formatted configuration files on the command line
set(HPX_COMMAND_LINE_HANDLING_WITH_JSON_CONFIGURATION_FILES_DEFAULT OFF)
if(HPX_COMMAND_LINE_HANDLING_WITH_JSON_CONFIGURATION_FILES)
  set(HPX_COMMAND_LINE_HANDLING_WITH_JSON_CONFIGURATION_FILES_DEFAULT
      ${HPX_COMMAND_LINE_HANDLING_WITH_JSON_CONFIGURATION_FILES}
  )
endif()

hpx_option(
  HPX_COMMAND_LINE_HANDLING_LOCAL_WITH_JSON_CONFIGURATION_FILES
  BOOL
  "Enable reading JSON formatted configuration files on the command line.\n
  (default: ${HPX_COMMAND_LINE_HANDLING_WITH_JSON_CONFIGURATION_FILES_DEFAULT})"
  ${HPX_COMMAND_LINE_HANDLING_WITH_JSON_CONFIGURATION_FILES_DEFAULT}
  ADVANCED
  CATEGORY "Modules"
  MODULE COMMAND_LINE_HANDLING_LOCAL
)

if(HPX_COMMAND_LINE_HANDLING_LOCAL_WITH_JSON_CONFIGURATION_FILES)
  hpx_add_config_define_namespace(
    DEFINE HPX_COMMAND_LINE_HANDLING_HAVE_JSON_CONFIGURATION_FILES
    NAMESPACE COMMAND_LINE_HANDLING_LOCAL
  )
endif()

set(command_line_handling_local_headers
    hpx/command_line_handling_local/command_line_handling_local.hpp
    hpx/command_line_handling_local/late_command_line_handling_local.hpp
    hpx/command_line_handling_local/parse_command_line_local.hpp
)

set(command_line_handling_local_sources
    command_line_handling_local.cpp late_command_line_handling_local.cpp
    parse_command_line_local.cpp
)

if(HPX_COMMAND_LINE_HANDLING_LOCAL_WITH_JSON_CONFIGURATION_FILES)
  include(HPX_SetupJSON)
  set(command_line_handling_local_dependencies Json::json)

  set(command_line_handling_local_headers
      ${command_line_handling_local_headers}
      hpx/command_line_handling_local/json_config_file.hpp
  )
  set(command_line_handling_local_sources ${command_line_handling_local_sources}
                                          json_config_file.cpp
  )
endif()

include(HPX_AddModule)
add_hpx_module(
  core command_line_handling_local
  GLOBAL_HEADER_GEN ON
  SOURCES ${command_line_handling_local_sources}
  HEADERS ${command_line_handling_local_headers}
  DEPENDENCIES ${command_line_handling_local_dependencies}
  MODULE_DEPENDENCIES
    hpx_config
    hpx_program_options
    hpx_runtime_configuration
    hpx_asio
    hpx_debugging
    hpx_errors
    hpx_filesystem
    hpx_format
    hpx_string_util
    hpx_topology
    hpx_util
    hpx_version
  CMAKE_SUBDIRS examples tests
)
