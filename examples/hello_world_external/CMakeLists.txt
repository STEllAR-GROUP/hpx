# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This is a top level CMakeLists.txt file for projects external to HPX
# which need to link to the HPX runtime system libraries. All of the
# CMake commands below are boilerplate ensuring HPX to be properly
# integrated. Usually there is no need to change anything except for
# the project name in the project() statement and the list of subdirectories
# to include.

# When invoking cmake you need to pass along the root directory of where you
# HPX installation is located:
#
#    cmake -DHPX_ROOT=<path_to_hpx_install> ...
#
# This will set up all required dependencies to the HPX headers, libraries and
# binaries.

project(hello_world_external CXX)

# This script will recurse into all specified subdirectories, and invoked the
# CMakeLists.txt files specified there. Add the names of all subdirectories
# you want to include into this project.

set(subdirs
    hello_world_client)


###############################################################################
# Do not change anything from this point on.

cmake_minimum_required(VERSION 2.8)

# Find and include HPX settings
if(NOT HPX_ROOT)
  message(FATAL "HPX_ROOT not set, unable to find HPX!")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${HPX_ROOT}/share/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}/Modules)

find_package(HPX)

# Add settings related to HPX to this project
include_directories(${HPX_INCLUDE_DIR})
link_directories(${HPX_LIBRARY_DIR})

# Adjust auxiliary build settings for HPX to work properly
if(NOT MSVC)
  set(HPX_RPATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}${HPX_RPATH})
else()
  # enable solution folders for MSVC
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  if(HPX_NO_INSTALL)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/$(Configuration)" CACHE PATH
      "Prefix prepended to install directories." FORCE)
  endif()
endif()

# just recurse into all subdirectories and execute the scripts there
foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()

