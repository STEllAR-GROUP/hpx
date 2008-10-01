# Copyright (c) 2007-2008 Hartmut Kaiser
# Copyright (c) 2007-2008 Chirag Dekate
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least CMake V2.6
cmake_minimum_required(VERSION 2.6)

# The name of our project is "hpx".  CMakeLists files in this project can
# refer to the root source directory of the project as ${hpx_SOURCE_DIR} and
# to the root binary directory of the project as ${hpx_BINARY_DIR}.
project (hpx CXX C)

# We have a patched BoostFIND module, look for it in our misc directory
set(CMAKE_MODULE_PATH ${hpx_SOURCE_DIR}/misc)

# We need to use Boost, namely the following Boost libraries
set(Boost_COMPONENTS_NEEDED
    date_time
    filesystem 
    graph 
    regex 
    serialization 
    thread 
    system 
    signals
)
set(Boost_USE_STATIC_LIBS OFF)
find_package(Boost 1.36.0 COMPONENTS ${Boost_COMPONENTS_NEEDED})

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# We need Intels Thread Building Blocks library as well
#find_package(TBB)

# We need to include the hpx directory for includes
include_directories(${hpx_SOURCE_DIR})
include_directories(${hpx_SOURCE_DIR}/external/coroutine)
include_directories(${hpx_SOURCE_DIR}/external/endian)
include_directories(${hpx_SOURCE_DIR}/external/exception)
include_directories(${hpx_SOURCE_DIR}/external/logging)
include_directories(${hpx_SOURCE_DIR}/external/lockfree)
include_directories(${hpx_SOURCE_DIR}/external/plugin)
include_directories(${hpx_SOURCE_DIR}/external/singleton)

# initialize installation directory (can be changed via ccmake or 
# 'cmake -DCMAKE_PREFIX=...')
if(UNIX)
    SET(CMAKE_PREFIX "/usr/lib" CACHE PATH "Prefix prepended to install directories")
else(UNIX)
    SET(CMAKE_PREFIX "C:/Program Files/hpx" CACHE PATH "Prefix prepended to install directories")
endif(UNIX)

set(CMAKE_INSTALL_PREFIX "${CMAKE_PREFIX}" CACHE INTERNAL "Prefix prepended to install directories" FORCE)

# global library configuration
add_definitions(-DHPX_PREFIX=\"${CMAKE_INSTALL_PREFIX}\")
add_definitions(-DHPX_USE_LOCKFREE=1)
add_definitions(-DHPX_USE_TBB=0)

# the Boost serialization library needs to be linked as a shared library
add_definitions(-DBOOST_SERIALIZATION_DYN_LINK)
add_definitions(-DBOOST_ARCHIVE_DYN_LINK)

# all other Boost libraries may be loaded as shared libraries
add_definitions(-DBOOST_FILESYSTEM_DYN_LINK)
add_definitions(-DBOOST_DATE_TIME_DYN_LINK)
add_definitions(-DBOOST_THREAD_DYN_DLL)
add_definitions(-DBOOST_REGEX_DYN_LINK)
add_definitions(-DBOOST_SYSTEM_DYN_LINK)

# compiler/platform specific configuration
if(MSVC)
    add_definitions(-D_WINDOWS)
    add_definitions(-DBOOST_USE_WINDOWS_H)
    add_definitions(-D_WIN32_WINNT=0x0501)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SCL_SECURE_NO_DEPRECATE)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif(MSVC)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-DHPX_GCC_HAVE_VISIBILITY)
endif(CMAKE_COMPILER_IS_GNUCXX)

# hpx_LIBRARIES lists all libraries a HPX module needs to be linked with by 
# default
set(hpx_LIBRARIES hpx hpx_serialization)

# Recurse into some subdirectories. This does not actually cause another cmake 
# executable to run. The same process will walk through the project's entire 
# directory structure.
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(runtime)

# Components are to build separately
add_subdirectory(hpx/components/accumulator)
add_subdirectory(hpx/components/simple_accumulator)
add_subdirectory(hpx/components/distributing_factory)

