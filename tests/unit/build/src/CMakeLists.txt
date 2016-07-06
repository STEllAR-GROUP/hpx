# Copyright (c) 2014-2016 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 2.8)

project(test CXX)

if(EXISTS "${HPX_DIR}")

  find_package(HPX REQUIRED)

  if(NOT HPX_FOUND)
    message(FATAL_ERROR "HPX could not be found")
  endif()

  include_directories(${test_SOURCE_DIR})
  include_directories(${HPX_INCLUDE_DIRS})
  link_directories(${HPX_LIBRARY_DIR})

  add_library(test_component SHARED test_component.cpp)
  hpx_setup_target(
    test_component
    TYPE COMPONENT
  )

  add_executable(test_executable test.cpp)
  target_link_libraries(test_executable ${HPX_LIBRARIES} iostreams_component test_component)
  target_compile_definitions(test_executable PUBLIC -DHPX_DEBUG)
  hpx_setup_target(test_executable
    DEPENDENCIES test_component
    COMPONENT_DEPENDENCIES iostreams
  )
else()
  message(WARNING "HPX_DIR=${HPX_DIR} does not exist. Did you forget to run the install rule?")
endif()
