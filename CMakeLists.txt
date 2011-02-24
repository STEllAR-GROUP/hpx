# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least CMake V2.6.2
cmake_minimum_required(VERSION 2.6.2 FATAL_ERROR)

################################################################################
# project metadata
################################################################################
project(hpx CXX C)

set(HPX_MAJOR_VERSION 0)
set(HPX_MINOR_VERSION 5)
set(HPX_PATCH_LEVEL   0)
set(HPX_VERSION "${HPX_MAJOR_VERSION}.${HPX_MINOR_VERSION}.${HPX_PATCH_LEVEL}")
set(HPX_SOVERSION ${HPX_MAJOR_VERSION})

################################################################################
# cmake configuration
################################################################################
set(CMAKE_MODULE_PATH ${hpx_SOURCE_DIR}/cmake)

# include additional macro definitions
include(HPX_Utils)

hpx_force_out_of_tree_build("This project requires an out-of-source-tree build. See INSTALL.rst. Clean your CMake cache and CMakeFiles if this message persists.")

# allow more human readable "if then else" constructs
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)

# be pedantic and annoying by default
if(NOT HPX_CMAKE_LOGLEVEL)
  set(HPX_CMAKE_LOGLEVEL "WARN")
endif()

################################################################################
# environment detection 
################################################################################
execute_process(COMMAND "${hpx_SOURCE_DIR}/tools/build_env.py" "${CMAKE_CXX_COMPILER}"
                OUTPUT_VARIABLE build_environment
                OUTPUT_STRIP_TRAILING_WHITESPACE)

if(build_environment)
  set(BUILDNAME "${build_environment}" CACHE INTERNAL "A string describing the build environment.")

  hpx_info("build_env" "Build environment is ${BUILDNAME}")
else()
  hpx_warn("build_env" "Couldn't determine build environment (install python).") 
endif()

################################################################################
# Boost configuration
################################################################################
# Boost.Chrono is in the Boost trunk, but has not been in a Boost release yet

option(HPX_INTERNAL_CHRONO "Use HPX's internal version of Boost.Chrono (default: ON for Boost < 1.47)" ON)

# this cmake module will snag the Boost version we'll be using (which we need
# to know to specify the Boost libraries that we want to look for).
find_package(HPX_BoostVersion)

if(NOT HPX_INTERNAL_CHRONO OR ${BOOST_MINOR_VERSION} GREATER 46)
  set(BOOST_LIBRARIES chrono
                      date_time
                      filesystem
                      program_options
                      regex
                      serialization
                      system
                      signals
                      thread)
  add_definitions(-DHPX_CHRONO_DONT_USE_INTERNAL_VERSION)
else()
  set(BOOST_LIBRARIES date_time
                      filesystem
                      program_options
                      regex
                      serialization
                      system
                      signals
                      thread)
  add_definitions(-DBOOST_CHRONO_NO_LIB)
  include_directories(${hpx_SOURCE_DIR}/external/chrono)
endif()

# We have a patched version of FindBoost loosely based on the one that Kitware ships
find_package(HPX_Boost)

include_directories(${BOOST_INCLUDE_DIR})
link_directories(${BOOST_LIB_DIR})

# the Boost serialization library needs to be linked as a shared library
add_definitions(-DBOOST_SERIALIZATION_DYN_LINK)
add_definitions(-DBOOST_ARCHIVE_DYN_LINK)

# all other Boost libraries don't need to be loaded as shared libraries
# (but it's easier configuration wise to do so)
add_definitions(-DBOOST_FILESYSTEM_DYN_LINK)
add_definitions(-DBOOST_DATE_TIME_DYN_LINK)
add_definitions(-DBOOST_PROGRAM_OPTIONS_DYN_LINK)
add_definitions(-DBOOST_REGEX_DYN_LINK)
add_definitions(-DBOOST_SYSTEM_DYN_LINK)
add_definitions(-DBOOST_SIGNALS_DYN_LINK)
add_definitions(-DBOOST_THREAD_DYN_DLL)

# additional preprocessor definitions (TODO: find out what these do)
add_definitions(-DBOOST_COROUTINE_USE_ATOMIC_COUNT) 
add_definitions(-DBOOST_COROUTINE_ARG_MAX=2)

################################################################################
# include configuration
################################################################################
include_directories(${hpx_SOURCE_DIR})
include_directories(${hpx_SOURCE_DIR}/external/atomic)
include_directories(${hpx_SOURCE_DIR}/external/cache)
include_directories(${hpx_SOURCE_DIR}/external/coroutine)
include_directories(${hpx_SOURCE_DIR}/external/endian)
include_directories(${hpx_SOURCE_DIR}/external/logging)
include_directories(${hpx_SOURCE_DIR}/external/lockfree)
include_directories(${hpx_SOURCE_DIR}/external/plugin)

################################################################################
# installation configuration
################################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

if(UNIX)
  # on *nix, we do a local system install 
  set(CMAKE_PREFIX "/usr/local" CACHE PATH "Prefix prepended to install directories.")
else()
  set(CMAKE_PREFIX "C:/Program Files/hpx" CACHE PATH "Prefix prepended to install directories.")
endif()

set(CMAKE_INSTALL_PREFIX "${CMAKE_PREFIX}"
  CACHE PATH "Where to install ${PROJECT_NAME} (default: /usr/local/ for POSIX, C:/Program Files/hpx for Windows)." FORCE)

hpx_info("install" "Install root is ${CMAKE_PREFIX}")

add_definitions(-DHPX_PREFIX=\"${CMAKE_INSTALL_PREFIX}\")

################################################################################
# rpath configuration
################################################################################
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

################################################################################
# hardware support
################################################################################
option(HPX_HW_ACCELERATION "Compile in support for hardware acceleration (default: off)." OFF)

if(HPX_HW_ACCELERATION)
  find_package(HPX_PXACCEL)

  if(PXACCEL_FOUND)
    add_definitions(-DHPX_ACCEL_QUEUING)
    include_directories(${PXACCEL_INC_DIR})
    link_directories(${PXACCEL_LIB_DIR})
    set(pxaccel_LIBRARIES pciutil pciaccess)
  endif()
endif()

################################################################################
# Windows specific configuration 
################################################################################
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

################################################################################
# POSIX specific configuration 
################################################################################
else()
  ##############################################################################
  # GNU specific configuration
  ##############################################################################
  option(HPX_ELF_HIDDEN_VISIBILITY
    "Use -fvisibility=hidden for Release, MinSizeRel and RelWithDebInfo builds (GNU GCC only, default: ON)" ON)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if(HPX_ELF_HIDDEN_VISIBILITY)
      add_definitions(-DHPX_GCC_HAVE_VISIBILITY)
      add_definitions(-DBOOST_COROUTINE_GCC_HAVE_VISIBILITY)
      set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -fvisibility=hidden")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fvisibility=hidden")
      set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fvisibility=hidden")
      set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} -fvisibility=hidden")
      set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fvisibility=hidden")
      set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fvisibility=hidden")
    endif()
  ##############################################################################
  # Intel specific configuration
  ##############################################################################
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # warning #191: type qualifier is meaningless on cast type
    add_definitions(-diag-disable 191) 
    
    # warning #279: controlling expression is constant 
    add_definitions(-diag-disable 279) 
    
    # warning #68: integer conversion resulted in a change of sign 
    add_definitions(-diag-disable 68) 
    
    # warning #858: type qualifier on return type is meaningless 
    add_definitions(-diag-disable 858) 
    
    # warning #1125: virtual function override intended 
    add_definitions(-diag-disable 1125) 

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ipo")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ipo")
    set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -ipo")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ipo")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ipo")
  endif()

  hpx_check_pthreads_setaffinity_np(hpx_HAVE_PTHREAD_SETAFFINITY_NP)

  if(hpx_HAVE_PTHREAD_SETAFFINITY_NP)
    add_definitions(-DHPX_HAVE_PTHREAD_SETAFFINITY_NP)
  endif()
  
  set(hpx_LIBRARIES ${hpx_LIBRARIES} dl pthread rt)
endif()

################################################################################
# Mac OS X specific configuration 
################################################################################
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  add_definitions(-D_XOPEN_SOURCE=1) # for some reason Darwin whines without this
endif()

################################################################################
# target specification 
################################################################################
# hpx_LIBRARIES lists all libraries a HPX module needs to be linked with by 
# default, there may be more for individual modules, but this is the mandatory set
set(hpx_LIBRARIES ${hpx_LIBRARIES} hpx hpx_serialization)

# Recurse into some subdirectories. This does not actually cause another cmake 
# executable to run. The same process will walk through the project's entire 
# directory structure.
add_subdirectory(src)

# components are to be built separately
add_subdirectory(hpx/components)

# executables that depend on libraries above
add_custom_target(examples) 
add_subdirectory(examples)

# TODO: bring this into the CMake tree
#add_custom_target(applications WORKING_DIRECTORY ${hpx_BINARY_DIR}/applications)
#add_subdirectory(applications EXCLUDE_FROM_ALL)

add_subdirectory(runtime)

################################################################################
# test configuration 
################################################################################
enable_testing()
include(CTest)
add_custom_target(tests) 
add_subdirectory(tests)

################################################################################
# installation instructions
################################################################################
install( # install all hpx header files
  DIRECTORY hpx/ 
  DESTINATION include/hpx 
  COMPONENT core
  FILES_MATCHING PATTERN "*.h*"
  PATTERN ".svn" EXCLUDE
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE)

install( # install all hpx cmake utility files
  DIRECTORY cmake/ 
  DESTINATION share/cmake 
  COMPONENT core
  FILES_MATCHING PATTERN "*.cmake"
  PATTERN ".svn" EXCLUDE
  PATTERN ".git" EXCLUDE)

install( # install all hpx python scripts
  DIRECTORY tools 
  DESTINATION share/hpx/python 
  COMPONENT core
  FILES_MATCHING PATTERN "*.py"
  PATTERN ".svn" EXCLUDE
  PATTERN ".git" EXCLUDE)

install( # install external dependencies
  DIRECTORY external/cache/boost
            external/coroutine/boost
            external/atomic/boost
            external/endian/boost
            external/lockfree/boost
            external/logging/boost
            external/plugin/boost
  DESTINATION include
  COMPONENT core
  FILES_MATCHING PATTERN "*.h*"
  PATTERN ".svn" EXCLUDE
  PATTERN ".git" EXCLUDE)

if(HPX_INTERNAL_CHRONO AND ${BOOST_MINOR_VERSION} LESS 47)
  install( # install Boost.Chrono headers if we're using our internal version
    DIRECTORY external/chrono/boost
    DESTINATION include
    COMPONENT core
    FILES_MATCHING PATTERN "*.h*"
    PATTERN ".svn" EXCLUDE
    PATTERN ".git" EXCLUDE)
endif()

