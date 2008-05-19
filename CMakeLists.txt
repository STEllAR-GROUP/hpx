# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least CMake V2.6
cmake_minimum_required(VERSION 2.6)

# The name of our project is "hpx".  CMakeLists files in this project can
# refer to the root source directory of the project as ${hpx_SOURCE_DIR} and
# to the root binary directory of the project as ${hpx_BINARY_DIR}.
project (hpx)

# We need to use Boost
find_package(Boost REQUIRED)

# We need to include the hpx directory for includes
include_directories (${hpx_SOURCE_DIR})
include_directories (${hpx_SOURCE_DIR}/external/endian)
include_directories (${hpx_SOURCE_DIR}/external/logging)
include_directories (${hpx_SOURCE_DIR}/external/singleton)

if(WIN32)
    add_definitions(-D_WIN32_WINNT=0x0501)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SCL_SECURE_NO_DEPRECATE)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif(WIN32)

# Recurse into the "src" and "test" subdirectories.  This does not actually
# cause another cmake executable to run.  The same process will walk through
# the project's entire directory structure.
add_subdirectory (src)
add_subdirectory (examples)

