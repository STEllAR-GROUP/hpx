# Copyright (c) 2018 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# We require at least CMake V3.3.2
cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

project(HPX.assert CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(HPX_ASSERT_WITH_TESTS "Include tests for assert" On)

message(STATUS "assert: Configuring")

add_subdirectory(examples)
add_subdirectory(src)
add_subdirectory(tests)

message(STATUS "assert: Configuring done")
