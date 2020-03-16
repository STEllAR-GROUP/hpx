# Copyright (c) 2014-2016 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13)

project(hello_world_client CXX)

if(EXISTS "${HPX_DIR}")
  find_package(HPX REQUIRED)

  add_library(hello_world_component SHARED hello_world_component.cpp)

  add_executable(hello_world_client hello_world_client.cpp)
  target_include_directories(hello_world_client PRIVATE ${test_SOURCE_DIR})
  target_include_directories(hello_world_client PRIVATE HPX_DEBUG)

  if("${SETUP_TYPE}" STREQUAL "TARGETS")
    target_link_libraries(hello_world_component PUBLIC HPX::hpx HPX::iostreams_component)
    target_link_libraries(hello_world_component PRIVATE HPX::component)

    target_link_libraries(hello_world_client PRIVATE hello_world_component)
  elseif("${SETUP_TYPE}" STREQUAL "MACROS")
    hpx_setup_target(hello_world_component
      COMPONENT_DEPENDENCIES iostreams
      TYPE COMPONENT)

    hpx_setup_target( hello_world_client
      DEPENDENCIES hello_world_component)
  else()
    message(FATAL_ERROR "Unknown SETUP_TYPE=\"${SETUP_TYPE}\"")
  endif()
else()
  message(WARNING "HPX_DIR=${HPX_DIR} does not exist. Did you forget to run the install rule?")
endif()
