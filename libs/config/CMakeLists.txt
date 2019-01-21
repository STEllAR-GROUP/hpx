# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# We require at least CMake V3.3.2
cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

project(HPX.config CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(HPX_CONFIG_WITH_TESTS "Include tests for config" On)

message(STATUS "config: Configuring")

add_subdirectory(examples)
add_subdirectory(src)
add_subdirectory(tests)

install(
  DIRECTORY include/hpx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpx
  COMPONENT config)

message(STATUS "config: Configuring done")
