# Copyright (c) 2026 Pratyksh Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.18)
project(test_package CXX)

find_package(HPX REQUIRED)

add_executable(test_package test_package.cpp)
target_link_libraries(test_package HPX::hpx)

# Set C++ standard to match HPX requirements
set_target_properties(
  test_package
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
)
