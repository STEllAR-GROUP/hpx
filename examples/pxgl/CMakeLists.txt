# Copyright (c) 2010-2011 Dylan Stark
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Require CMake version >= 2.6.4
cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

if(NOT DEFINED HPX_ROOT)
  set(HPX_ROOT $ENV{HPX_ROOT})
endif()

set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
    "${HPX_ROOT}/share/hpx/cmake/HPX_CXXOverrides.cmake")

project(pxgl CXX)

include(${HPX_ROOT}/share/hpx/cmake/HPX.cmake)

################################################################################
# Setup PXGL
include_directories(${pxgl_SOURCE_DIR})

add_subdirectory(src)
add_subdirectory(examples)

# Install default INI in build directory. This allows you to always run
# example codes from the build directory without having to manually copy
# the default ini file.
install(FILES ${pxgl_SOURCE_DIR}/misc/.hpx.ini 
        DESTINATION ${pxgl_BINARY_DIR})

