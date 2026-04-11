# Copyright (c) 2025 Pratyksh Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.20)
project(pkgconfig_generation_test NONE)

set(HPX_COMPONENTS "")
set(CMAKE_INSTALL_LIBDIR "lib")
set(CMAKE_FILES_DIRECTORY "/CMakeFiles")
set(PROJECT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(HPX_SOURCE_DIR "${HPX_SOURCE_DIR}")

macro(hpx_handle_component_dependencies)

endmacro()

include("${HPX_SOURCE_DIR}/cmake/HPX_GeneratePackageUtils.cmake")

add_library(hpx_mock INTERFACE)
set_target_properties(
  hpx_mock PROPERTIES INTERFACE_COMPILE_DEFINITIONS "HPX_MOCK_TEST"
                      INTERFACE_INCLUDE_DIRECTORIES "/usr/include/hpx_mock"
)

hpx_generate_pkgconfig_from_target(hpx_mock hpx_application TRUE)
hpx_generate_pkgconfig_from_target(hpx_mock hpx_application FALSE)

get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(config_types ${CMAKE_CONFIGURATION_TYPES})
if(NOT config_types)
  set(config_types Debug Release RelWithDebInfo MinSizeRel)
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LC)

set(VERIFY_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/verify_pc.cmake")
file(
  WRITE "${VERIFY_SCRIPT}"
  "
  set(is_multi_config ${is_multi_config})
  set(config_types \"${config_types}\")
  set(CMAKE_BUILD_TYPE_LC \"${CMAKE_BUILD_TYPE_LC}\")

  set(BUILD_PC_DIR \"${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig\")
  set(INSTALL_PC_DIR \"${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles\")

  macro(check_dir dir_path)
    if(is_multi_config)
      foreach(config \${config_types})
        string(TOLOWER \${config} config_lc)
        set(pc_file \"\${dir_path}/hpx_application_\${config_lc}.pc\")
        if(NOT EXISTS \"\${pc_file}\")
          message(FATAL_ERROR \"Multi-config: Failed to generate pkg-config file for \${config}: \${pc_file}\")
        endif()

        file(READ \"\${pc_file}\" content)
        if(NOT content MATCHES \"HPX_MOCK_TEST\")
           message(FATAL_ERROR \"Content verification failed for \${pc_file}: Missing HPX_MOCK_TEST\")
        endif()
        message(STATUS \"Verified: \${pc_file}\")
      endforeach()

      if(EXISTS \"\${dir_path}/hpx_application.pc\")
         file(READ \"\${dir_path}/hpx_application.pc\" content)
         if(NOT content MATCHES \"HPX_MOCK_TEST\")
            message(FATAL_ERROR \"Content verification failed for legacy \${dir_path}/hpx_application.pc\")
         endif()
         message(STATUS \"Verified legacy: hpx_application.pc in \${dir_path}\")
      endif()
    else()
      set(pc_file \"\${dir_path}/hpx_application_\${CMAKE_BUILD_TYPE_LC}.pc\")
      if(NOT EXISTS \"\${pc_file}\")
         message(FATAL_ERROR \"Single-config: Failed to generate pkg-config file for \${CMAKE_BUILD_TYPE_LC}: \${pc_file}\")
      endif()

      file(READ \"\${pc_file}\" content)
      if(NOT content MATCHES \"HPX_MOCK_TEST\")
         message(FATAL_ERROR \"Content verification failed for \${pc_file}\")
      endif()
      message(STATUS \"Verified: \${pc_file}\")

      if(\"\${CMAKE_BUILD_TYPE_LC}\" MATCHES \"rel\")
         if(EXISTS \"\${dir_path}/hpx_application.pc\")
            message(STATUS \"Verified legacy: hpx_application.pc in \${dir_path}\")
         else()
            message(FATAL_ERROR \"Legacy pkg-config file missing for release-like build in \${dir_path}\")
         endif()
      endif()
    endif()
  endmacro()

  message(STATUS \"Checking build directory pkg-config files...\")
  check_dir(\"\${BUILD_PC_DIR}\")
  message(STATUS \"Checking simulated install directory pkg-config files...\")
  check_dir(\"\${INSTALL_PC_DIR}\")
"
)

add_custom_target(check_pc ALL COMMAND ${CMAKE_COMMAND} -P "${VERIFY_SCRIPT}")
