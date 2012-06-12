
#  Copyright (c) 2012 Thomas Heller
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http:#www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required (VERSION 2.6)

project(OCLM)

# set cmake module path to find our custom cmake scripts
set(CMAKE_MODULE_PATH "${OCLM_SOURCE_DIR}/cmake")

include(OCLM_AddLibrary)
include(OCLM_Message)

include_directories(${OCLM_SOURCE_DIR}/include)

add_definitions(-DOCLM_SOURCE_DIR=\"${OCLM_SOURCE_DIR}\")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ansi")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")

find_package(OpenCL REQUIRED)
oclm_add_library(${OPENCL_LIBRARIES})

link_directories(${OPENCL_LIBPATH})

find_package(GLEW)
if(GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIR})
    add_definitions(-DOCL_GL_INTEROP)
    oclm_add_library(${GLEW_LIBRARIES})
endif()

# This cmake module will snag the Boost version we'll be using (which we need
# to know to specify the Boost libraries that we want to look for)
find_package(OCLM_BoostVersion)

if(NOT BOOST_VERSION_FOUND)
  oclm_error("boost" "Failed to locate Boost.")
endif()

###############################################################################
# Boost libraries which are always used from the installed version
set(BOOST_LIBRARIES date_time
                    filesystem
                    program_options
                    regex
                    serialization
                    system
                    thread)

# We have a patched version of FindBoost loosely based on the one that Kitware ships
find_package(OCLM_Boost)

link_directories(${BOOST_LIB_DIR})
oclm_add_library(${BOOST_program_options_LIBRARY})
oclm_add_library(${BOOST_thread_LIBRARY})
oclm_add_library(${BOOST_regex_LIBRARY})
oclm_add_library(${BOOST_system_LIBRARY})
oclm_add_library(${BOOST_filesystem_LIBRARY})
include_directories(${BOOST_INCLUDE_DIR})
  
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(${OCLM_SOURCE_DIR}/src)
add_subdirectory(${OCLM_SOURCE_DIR}/examples)
