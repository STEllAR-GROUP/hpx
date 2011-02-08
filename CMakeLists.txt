# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
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
set(HPX_MINOR_VERSION 5)
set(HPX_PATCH_LEVEL   0)
set(HPX_VERSION "${HPX_MAJOR_VERSION}.${HPX_MINOR_VERSION}.${HPX_PATCH_LEVEL}")
set(HPX_SOVERSION ${HPX_MAJOR_VERSION})

# We have a patched BoostFIND module, and other files to include, look for 
# these in our misc directory
list(APPEND CMAKE_MODULE_PATH ${hpx_SOURCE_DIR}/misc)
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

# include additional macro definitions
include(HpxUtils)
include(FindBoost)

find_boost_library(date_time)
find_boost_library(filesystem)
find_boost_library(program_options)
find_boost_library(regex)
find_boost_library(serialization)
find_boost_library(system)
find_boost_library(signals)
find_boost_library(thread)

if(NOT HPX_INTERNAL_CHRONO)
  set(HPX_INTERNAL_CHRONO ON
      CACHE BOOL "Use HPX's internal version of Boost.Chrono (default: ON for Boost < 1.47)")
else()
  set(HPX_INTERNAL_CHRONO ${HPX_INTERNAL_CHRONO}
      CACHE BOOL "Use HPX's internal version of Boost.Chrono (default: ON for Boost < 1.47).")
endif()

# Chrono is not in 1.46 release
if(NOT HPX_INTERNAL_CHRONO OR ${BOOST_MINOR_VERSION} GREATER 46)
    find_boost_library(chrono)
endif()

include_directories(${BOOST_INCLUDE_DIR})
link_directories(${BOOST_LIB_DIR})

# set default cmake build type to RelWithDebInfo (None Debug Release RelWithDebInfo MinSizeRel)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

# Enable testing
if("$ENV{HPX_TESTS}" STREQUAL "all")
    #set(CMAKE_SKIP_RPATH TRUE)
    ENABLE_TESTING()
endif("$ENV{HPX_TESTS}" STREQUAL "all")

# We need Intels Thread Building Blocks library as well
#find_package(TBB)

# We need to include several hpx directories for header files
include_directories(${hpx_SOURCE_DIR})
include_directories(${hpx_SOURCE_DIR}/external/atomic)
include_directories(${hpx_SOURCE_DIR}/external/cache)
if(HPX_INTERNAL_CHRONO AND ${BOOST_MINOR_VERSION} LESS 47)
    include_directories(${hpx_SOURCE_DIR}/external/chrono)
endif()
include_directories(${hpx_SOURCE_DIR}/external/coroutine)
include_directories(${hpx_SOURCE_DIR}/external/endian)
if(${BOOST_MINOR_VERSION} LESS 36)
    include_directories(${hpx_SOURCE_DIR}/external/exception)
endif()
include_directories(${hpx_SOURCE_DIR}/external/logging)
include_directories(${hpx_SOURCE_DIR}/external/lockfree)
include_directories(${hpx_SOURCE_DIR}/external/plugin)

# initialize installation directory (can be changed via ccmake or 
# 'cmake -DCMAKE_PREFIX=...')
if(UNIX)
    SET(CMAKE_PREFIX "/usr/local" 
        CACHE PATH "Prefix prepended to install directories")
else()
    SET(CMAKE_PREFIX "C:/Program Files/hpx" 
        CACHE PATH "Prefix prepended to install directories")
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

set(CMAKE_SKIP_BUILD_RPATH FALSE)

set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

#if(BOOST_ROOT AND NOT BOOST_LIB_DIR AND NOT BOOST_USE_SYSTEM)
#  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} "${BOOST_ROOT}/stage/lib")
#elseif(BOOST_LIB_DIR AND NOT BOOST_USE_SYSTEM)
#  set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} "${BOOST_LIB_DIR}")
#endif()

set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} "${CMAKE_INSTALL_PREFIX}/lib")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

message(STATUS "HPX will be installed to: " ${CMAKE_INSTALL_PREFIX})

# top-level defines related to hardware support
set(HPX_HW_ACCELERATION 0
    CACHE BOOL "Compile in support for hardware acceleration")
mark_as_advanced(HPX_HW_ACCELERATION)

# global library configuration
if(NOT HPX_INTERNAL_CHRONO)
    add_definitions(-DHPX_CHRONO_DONT_USE_INTERNAL_VERSION)
endif()

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

# Boost.Chrono gets linked in directly (at least for now)
if(HPX_INTERNAL_CHRONO AND ${BOOST_MINOR_VERSION} LESS 47)
    add_definitions(-DBOOST_CHRONO_NO_LIB)
endif()

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
    add_definitions(-wd4251 -wd4231 -wd4275 -wd4660 -wd4094 -wd4267 -wd4180 -wd4244)
    if(CMAKE_CL_64)
        add_definitions(-DBOOST_COROUTINE_USE_FIBERS)
    endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR (XIAR AND UNIX)
    OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
    OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "PathScale")
    add_definitions(-DHPX_GCC_HAVE_VISIBILITY)
    add_definitions(-DBOOST_COROUTINE_GCC_HAVE_VISIBILITY)

    # check for availability of pthread_setaffinity_np()
    include(CheckSymbolExists)
    check_symbol_exists(pthread_setaffinity_np "pthread.h"
        HAVE_PTHREAD_SETAFFINITY_NP)
    if(HAVE_PTHREAD_SETAFFINITY_NP)
        add_definitions(-DHAVE_PTHREAD_SETAFFINITY_NP)
    endif()
endif()

# additional preprocessor definitions
add_definitions(-DBOOST_COROUTINE_USE_ATOMIC_COUNT)
add_definitions(-DBOOST_COROUTINE_ARG_MAX=2)

# Mac systems headers have some dependency on _XOPEN_SOURCE being defined
# let's better be on the safe side...
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    add_definitions(-D_XOPEN_SOURCE=1)
endif()

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

# hardware support, if available
if(HPX_HW_ACCELERATION)
    find_package(PXACCEL)
    if(PXACCEL_FOUND)
        message(STATUS "Hardware acceleration framework found in ${PXACCEL_ROOT}")
        add_definitions(-DHPX_ACCEL_QUEUING)
    include_directories(${PXACCEL_INC_DIR})
    set(pxaccel_LIBRARIES pciutil pciaccess)
    link_directories(${PXACCEL_LIB_DIR})
    else()
        message(FATAL ": Hardware acceleration support was requested, but the related access\n"
                "library was not found. Please make sure that PXACCEL_ROOT points to the top\n"
                "installation directory of the acceleration, framework, or unset the\n"
                "HPX_HW_ACCELERATION variable in cmake.")
    endif()
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
#add_subdirectory(applications)
add_subdirectory(runtime)

# installation instructions
install(
    DIRECTORY hpx/                    # install all hpx header files
    DESTINATION include/hpx 
    FILES_MATCHING PATTERN "*.h*"
    PATTERN ".svn" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE)
install(
    DIRECTORY misc/               # install all hpx build utility files
    DESTINATION share/cmake 
    FILES_MATCHING PATTERN "*.cmake"
    PATTERN "hpx_log" EXCLUDE
    PATTERN "pxdm" EXCLUDE
    PATTERN ".svn" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE)

# install external dependencies
install(
    DIRECTORY external/cache/boost
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE)
if(HPX_INTERNAL_CHRONO AND ${BOOST_MINOR_VERSION} LESS 47)
    install(
        DIRECTORY external/chrono/boost
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN ".svn" EXCLUDE)
endif()
install(
    DIRECTORY external/coroutine/boost
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE)
install(
    DIRECTORY external/atomic/boost
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE)
install(
    DIRECTORY external/endian/boost
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE)
if(${BOOST_MINOR_VERSION} LESS 36)
    install(
        DIRECTORY external/exception/boost
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp"
        PATTERN ".svn" EXCLUDE)
endif()
install(
    DIRECTORY external/lockfree/boost
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE)
install(
    DIRECTORY external/logging/boost
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE)
install(
    DIRECTORY external/plugin/boost
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE)

# export build settings
include(CMakeExportBuildSettings)
cmake_export_build_settings("${PROJECT_NAME}BuildSettings.cmake")

# export library dependencies (keep this as the last line in the file)
export_library_dependencies("${PROJECT_NAME}LibDeps.cmake")

