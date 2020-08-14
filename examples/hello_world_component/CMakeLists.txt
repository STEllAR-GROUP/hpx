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
    target_link_libraries(
      hello_world_component PUBLIC HPX::hpx HPX::iostreams_component
    )
    target_link_libraries(hello_world_component PRIVATE HPX::component)

    target_link_libraries(
      hello_world_client PRIVATE hello_world_component HPX::wrap_main
    )

    # We still support not linking to HPX::wrap_main when
    # HPX_WITH_DYNAMIC_HPX_MAIN=OFF for legacy use. This can only be done using
    # an installed HPX. It is strongly recommended to always link to
    # HPX::wrap_main when implicitly using main as the runtime entry point.
    if(USING_INSTALL_DIR AND NOT HPX_WITH_DYNAMIC_HPX_MAIN)
      add_executable(hello_world_client_only_hpx_init hello_world_client.cpp)
      target_include_directories(
        hello_world_client_only_hpx_init PRIVATE ${test_SOURCE_DIR}
      )
      target_include_directories(
        hello_world_client_only_hpx_init PRIVATE HPX_DEBUG
      )

      target_link_libraries(
        hello_world_client_only_hpx_init PRIVATE hello_world_component
      )
    endif()
  elseif("${SETUP_TYPE}" STREQUAL "MACROS")
    hpx_setup_target(
      hello_world_component
      COMPONENT_DEPENDENCIES iostreams
      DEPENDENCIES HPX::wrap_main
      TYPE COMPONENT
    )

    hpx_setup_target(hello_world_client DEPENDENCIES hello_world_component)
  else()
    message(FATAL_ERROR "Unknown SETUP_TYPE=\"${SETUP_TYPE}\"")
  endif()
else()
  message(
    WARNING
      "Skipping build test because HPX_DIR=${HPX_DIR} does not exist. This \
      last test requires HPX to be installed.  Did you forget to run the \
      install rule?"
  )
endif()
