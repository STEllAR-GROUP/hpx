# Copyright (c) 2007-2008 Hartmut Kaiser
# Copyright (c) 2007-2008 Chirag Dekate
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least CMake V2.6.2
cmake_minimum_required(VERSION 2.6.2 FATAL_ERROR)

# allow more human readable "if then else" constructs
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)

# The name of our project is "hpx".  CMakeLists files in this project can
# refer to the root source directory of the project as ${hpx_SOURCE_DIR} and
# to the root binary directory of the project as ${hpx_BINARY_DIR}.
project(hpx CXX C)

# project version
set(HPX_MAJOR_VERSION 0)
set(HPX_MINOR_VERSION 2)
set(HPX_PATCH_LEVEL   0)
set(HPX_VERSION "${HPX_MAJOR_VERSION}.${HPX_MINOR_VERSION}.${HPX_PATCH_LEVEL}")

# We have a patched BoostFIND module, and other files to include, look for 
# these in our misc directory
list(APPEND CMAKE_MODULE_PATH ${hpx_SOURCE_DIR}/misc)

# include additional macro definitions
include(HpxUtils)

# set default cmake build type to RelWithDebInfo (None Debug Release RelWithDebInfo MinSizeRel)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

# We need to use Boost, namely the following Boost libraries
set(Boost_FIND_VERSION_EXACT ON)
set(Boost_FIND_VERSION_MAJOR 1)
set(Boost_FIND_VERSION_MINOR 37)
set(Boost_FIND_VERSION_PATCH 0)
set(Boost_COMPONENTS_NEEDED
    date_time
    filesystem
    graph
    program_options
    regex
    serialization
    system
    signals
    thread)
set(Boost_USE_STATIC_LIBS OFF)

find_package(Boost 1.37.0 COMPONENTS ${Boost_COMPONENTS_NEEDED})

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# We need Intels Thread Building Blocks library as well
#find_package(TBB)

# We need to include several hpx directories for header files
include_directories(${hpx_SOURCE_DIR})
include_directories(${hpx_SOURCE_DIR}/external/cache)
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
    SET(CMAKE_PREFIX "/usr/lib" 
        CACHE PATH "Prefix prepended to install directories")
else()
    SET(CMAKE_PREFIX "C:/Program Files/hpx" 
        CACHE PATH "Prefix prepended to install directories")
endif()

# Mac systems headers have some dependency on XOPEN_SOURCE being defined
# let's better be on the safe side...
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    add_definitions(-DXOPEN_SOURCE=1)
endif()

# force some variables that could be defined in the command line to be written 
# to cache
set(CMAKE_INSTALL_PREFIX "${CMAKE_PREFIX}" 
    CACHE PATH "Where to install ${PROJECT_NAME}" 
    FORCE)
mark_as_advanced(CMAKE_INSTALL_PREFIX)
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" 
    CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." 
    FORCE)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" 
    CACHE PATH "Path to custom CMake Modules" 
    FORCE)

# global library configuration
add_definitions(-DHPX_PREFIX=\"${CMAKE_INSTALL_PREFIX}\")
add_definitions(-DHPX_USE_LOCKFREE=1)
add_definitions(-DHPX_USE_TBB=0)

# the Boost serialization library needs to be linked as a shared library
add_definitions(-DBOOST_SERIALIZATION_DYN_LINK)
add_definitions(-DBOOST_ARCHIVE_DYN_LINK)

# all other Boost libraries don't need to be loaded as shared libraries (but 
# it's easier configuration wise to do so)
add_definitions(-DBOOST_FILESYSTEM_DYN_LINK)
add_definitions(-DBOOST_DATE_TIME_DYN_LINK)
add_definitions(-DBOOST_PROGRAM_OPTIONS_DYN_LINK)
add_definitions(-DBOOST_REGEX_DYN_LINK)
add_definitions(-DBOOST_SYSTEM_DYN_LINK)
add_definitions(-DBOOST_SIGNALS_DYN_LINK)
add_definitions(-DBOOST_THREAD_DYN_DLL)

# compiler/platform specific configuration
if(MSVC)
    add_definitions(-D_WINDOWS)
    add_definitions(-DBOOST_USE_WINDOWS_H)
    add_definitions(-D_WIN32_WINNT=0x0501)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SCL_SECURE_NO_DEPRECATE)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
    add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)
    # suppress certain warnings
    add_definitions(-wd4251 -wd4231 -wd4275 -wd4660 -wd4094 -wd4267)
    if(CMAKE_CL_64)
        add_definitions(-DBOOST_COROUTINES_USE_FIBERS)
    endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-DHPX_GCC_HAVE_VISIBILITY)
endif()

# additionally needed preprocessor definitions
add_definitions(-DBOOST_COROUTINE_USE_ATOMIC_COUNT)

# hpx_LIBRARIES lists all libraries a HPX module needs to be linked with by 
# default, there may be more for individual modules, but this is the mandatory
# set
set(hpx_LIBRARIES hpx hpx_serialization)
if(UNIX)
    set(hpx_LIBRARIES ${hpx_LIBRARIES} dl)
endif()

# set variable helping in naming the target library names
if(MSVC)
    set(CMAKE_DEBUG_POSTFIX "d")
endif()
if(UNIX)
    set(component_LIBRARY_PREFIX "hpx_component_")
endif()

# Recurse into some subdirectories. This does not actually cause another cmake 
# executable to run. The same process will walk through the project's entire 
# directory structure.
add_subdirectory(src)

# Components are to build separately
add_subdirectory(hpx/components)

# executables depend on libraries above
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(runtime)

# installation instructions
install(
    DIRECTORY hpx/                    # install all hpx header files
    DESTINATION include/hpx 
    FILES_MATCHING PATTERN "*.hpp"
    REGEX ".svn" EXCLUDE
    REGEX "CMakeFiles" EXCLUDE)

# export build settings
include(CMakeExportBuildSettings)
cmake_export_build_settings("${PROJECT_NAME}BuildSettings.cmake")

# export library dependencies (keep this as the last line in the file)
export_library_dependencies("${PROJECT_NAME}LibDeps.cmake")
