# Copyright (c) 2014-2016 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2)

project(hello_world_client CXX)

if(EXISTS "${HPX_DIR}")

  find_package(HPX REQUIRED)

  if(NOT HPX_FOUND)
    message(FATAL_ERROR "HPX could not be found")
  endif()

  include_directories(${test_SOURCE_DIR})
  include_directories(${HPX_INCLUDE_DIRS})
  link_directories(${HPX_LIBRARY_DIR})

  add_library(hello_world_component SHARED hello_world_component.cpp)
  hpx_setup_target(
    hello_world_component
    DEPENDENCIES iostreams_component
    TYPE COMPONENT
  )

  add_executable(hello_world_client hello_world_client.cpp)
  target_compile_definitions(hello_world_client PUBLIC -DHPX_DEBUG)
  hpx_setup_target(hello_world_client
    DEPENDENCIES hello_world_component
    COMPONENT_DEPENDENCIES iostreams
  )
else()
  message(WARNING "HPX_DIR=${HPX_DIR} does not exist. Did you forget to run the install rule?")
endif()
