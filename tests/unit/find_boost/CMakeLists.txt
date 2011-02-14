# Copyright (c) 2010-2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 2.6.2 FATAL_ERROR)

set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)

list(APPEND CMAKE_MODULE_PATH ${HPX_ROOT}/misc)
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

include(FindBoost)

project(boost_link_test CXX)

find_boost_library(date_time)
find_boost_library(filesystem)
find_boost_library(program_options)
find_boost_library(regex)
find_boost_library(serialization)
find_boost_library(system)
find_boost_library(thread)

include_directories(${BOOST_INCLUDE_DIR})
link_directories(${BOOST_LIB_DIR})

add_executable(boost_link_test boost_link_test.cpp)

target_link_libraries(boost_link_test ${BOOST_FOUND_LIBRARIES})

