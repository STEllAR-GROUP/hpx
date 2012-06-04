# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least CMake V2.8.4
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

# Overrides must go before the project() statement, otherwise they are ignored.

################################################################################
# C++ overrides
################################################################################
set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
    "${CMAKE_SOURCE_DIR}/cmake/HPX_CXXOverrides.cmake")

################################################################################
# Fortran overrides
################################################################################
set(CMAKE_USER_MAKE_RULES_OVERRIDE_Fortran
    "${CMAKE_SOURCE_DIR}/cmake/HPX_FortranOverrides.cmake")

################################################################################
# Build type (needs to be handled before project command below)
################################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)" FORCE)
endif()

# Always force CMAKE_CONFIGURATION_TYPES to be the same as CMAKE_BUILD_TYPE
# (at least for now, until we figure out how to use multiple build types in
# the same project).
set(CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPE}" CACHE INTERNAL "Configuration types" FORCE)

################################################################################
# project metadata
################################################################################
project(hpx CXX)

set(HPX_MAJOR_VERSION 0)
set(HPX_MINOR_VERSION 9)
set(HPX_PATCH_LEVEL   0)
set(HPX_VERSION "${HPX_MAJOR_VERSION}.${HPX_MINOR_VERSION}.${HPX_PATCH_LEVEL}")
set(HPX_SOVERSION ${HPX_MAJOR_VERSION})

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

################################################################################
# CMake configuration
################################################################################
set(CMAKE_MODULE_PATH ${hpx_SOURCE_DIR}/cmake)

# include additional macro definitions
include(HPX_Utils)

include(HPX_Distclean)

hpx_force_out_of_tree_build("This project requires an out-of-source-tree build. See README.rst. Clean your CMake cache and CMakeFiles if this message persists.")

if(NOT HPX_CMAKE_LOGLEVEL)
  set(HPX_CMAKE_LOGLEVEL "WARN")
endif()

###############################################################################
# reconfigure Boost library information if configuration type has been
# changed since last configure
if(CMAKE_BUILD_TYPE AND NOT (CMAKE_BUILD_TYPE STREQUAL CMAKE_CONFIGURATION_TYPES))
  set(BOOST_VERSION_SEARCHED OFF CACHE INTERNAL "Found Boost version" FORCE)
  set(BOOST_SEARCHED OFF CACHE INTERNAL "Found Boost libraries" FORCE)
endif()

################################################################################
# Fortran compiler detection
################################################################################
include(HPX_FortranCompiler)

################################################################################
# search path configuration
################################################################################
hpx_include_directories(${hpx_SOURCE_DIR})
hpx_include_directories(${hpx_SOURCE_DIR}/external/atomic)
hpx_include_directories(${hpx_SOURCE_DIR}/external/cache)
hpx_include_directories(${hpx_SOURCE_DIR}/external/coroutine)
hpx_include_directories(${hpx_SOURCE_DIR}/external/endian)
hpx_include_directories(${hpx_SOURCE_DIR}/external/logging)
hpx_include_directories(${hpx_SOURCE_DIR}/external/lockfree)
hpx_include_directories(${hpx_SOURCE_DIR}/external/plugin)
hpx_include_directories(${hpx_SOURCE_DIR}/external/backtrace)
hpx_link_directories(${CMAKE_BINARY_DIR}/lib/hpx)

################################################################################
# Boost configuration
################################################################################

# Boost.Chrono is in newer versions of Boost
hpx_option(HPX_INTERNAL_CHRONO BOOL "Use HPX's internal version of Boost.Chrono (default: OFF)" OFF ADVANCED)

# Boost.Move is in newer versions of Boost
hpx_option(HPX_INTERNAL_MOVE BOOL "Use HPX's internal version of Boost.Move (default: OFF)" OFF ADVANCED)

# This cmake module will snag the Boost version we'll be using (which we need
# to know to specify the Boost libraries that we want to look for)
find_package(HPX_BoostVersion)

if(NOT BOOST_VERSION_FOUND)
  hpx_error("boost" "Failed to locate Boost.")
endif()

if(NOT MSVC AND ${BOOST_MINOR_VERSION} LESS 47)
  hpx_warn("boost" "Applications will segfault at shutdown with this version of Boost, please use 1.47.0 or higher.")
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

if(${BOOST_MINOR_VERSION} LESS 49)
  hpx_include_directories(${hpx_SOURCE_DIR}/external/serialization)
endif()

# Decide whether to use internal version of Boost.Chrono
if(${BOOST_MINOR_VERSION} LESS 47)
  hpx_warn("boost" "Using internal version of Boost.Chrono, setting HPX_INTERNAL_CHRONO=ON.")
  set(HPX_INTERNAL_CHRONO ON CACHE BOOL "Use HPX's internal version of Boost.Chrono (default: OFF)" FORCE)
endif()

if(${HPX_INTERNAL_CHRONO})
  add_definitions(-DHPX_INTERNAL_CHRONO)
  add_definitions(-DBOOST_CHRONO_NO_LIB)
  hpx_include_directories(${hpx_SOURCE_DIR}/external/chrono)
else()
  set(BOOST_LIBRARIES ${BOOST_LIBRARIES} chrono)
endif()

# Decide whether to use internal version of Boost.Move
if(${BOOST_MINOR_VERSION} LESS 48)
  hpx_warn("boost" "Using internal version of Boost.Move, setting HPX_INTERNAL_MOVE=ON.")
  set(HPX_INTERNAL_MOVE ON CACHE BOOL "Use HPX's internal version of Boost.Move (default: OFF)" FORCE)
endif()

if(${HPX_INTERNAL_MOVE})
  add_definitions(-DHPX_INTERNAL_MOVE)
  hpx_include_directories(${hpx_SOURCE_DIR}/external/move)
endif()

# We have a patched version of FindBoost loosely based on the one that Kitware ships
find_package(HPX_Boost)

hpx_include_sys_directories(${BOOST_INCLUDE_DIR})
hpx_link_sys_directories(${BOOST_LIBRARY_DIR})

# Boost preprocessor definitions
add_definitions(-DBOOST_PARAMETER_MAX_ARITY=7)
add_definitions(-DBOOST_COROUTINE_USE_ATOMIC_COUNT)
add_definitions(-DBOOST_COROUTINE_ARG_MAX=2)
if(NOT MSVC)
  add_definitions(-DBOOST_COROUTINE_NO_SEPARATE_CALL_SITES)
endif()
add_definitions(-DBOOST_LOG_NO_TSS)
add_definitions(-DBOOST_LOG_NO_TS)
add_definitions(-DBOOST_BIGINT_HAS_NATIVE_INT64)
add_definitions(-DBOOST_PLUGIN_PREFIX=hpx)

# disable usage of std::atomics in lockfree
add_definitions(-DBOOST_NO_0X_HDR_ATOMIC)

################################################################################
# Compiler detection code
################################################################################

# C++
hpx_include(GCCVersion)

if(GCC_VERSION AND NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
  hpx_info("gcc_config" "Compiler reports compatibility with GCC version ${GCC_VERSION_STR}")

  if(040400 GREATER ${GCC_VERSION})
    hpx_error("gcc_config" "GCC 4.4.x or higher is required.")
  endif()
elseif(MSVC)
  if(NOT (MSVC10 OR MSVC11))
    hpx_error("msvc_config" "MSVC x64 2010 or higher is required.")
  elseif(NOT CMAKE_CL_64)
    hpx_warn("msvc_config" "MSVC (32Bit) will compile but will fail running larger applications because of limitations in the Windows OS.")
  endif()
endif()

# Fortran
if(CMAKE_Fortran_COMPILER)
  hpx_include(GFortranVersion)
  if(GFORTRAN_VERSION)
    hpx_info("gfortran_config" "Compiler reports compatibility with gfortran version ${GFORTRAN_VERSION_STR}")
  endif()
endif()

hpx_include(CompilerFlags)

################################################################################
# SVN revision detection
################################################################################
include(HPX_SVNRevision)

################################################################################
# Installation configuration
################################################################################
# FIXME: Three separate doc strings are used for the *_PREFIX macros.

hpx_option(HPX_NO_INSTALL BOOL "Build hpx applications so that they can be run without installation (default: OFF)." OFF ADVANCED)

if(NOT HPX_NO_INSTALL)
  # for backwards compatibility
  if(CMAKE_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_PREFIX}")
  endif()

  if(NOT CMAKE_INSTALL_PREFIX)
    if(UNIX)
      set(CMAKE_INSTALL_PREFIX "/opt/hpx" CACHE PATH "Prefix prepended to install directories.")
    else()
      set(CMAKE_INSTALL_PREFIX "C:/Program Files/hpx" CACHE PATH "Prefix prepended to install directories.")
    endif()
  endif()

  set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}"
    CACHE PATH "Where to install ${PROJECT_NAME} (default: /opt/hpx for POSIX, C:/Program Files/hpx for Windows)." FORCE)

  hpx_info("install" "Install root is ${CMAKE_INSTALL_PREFIX}.")
endif()

################################################################################
# Logging configuration
################################################################################
hpx_option(HPX_NO_LOGGING BOOL "Build hpx with logging completely disabled (default: OFF)." OFF ADVANCED)
if(HPX_NO_LOGGING)
  add_definitions("-DHPX_NO_LOGGING=1")
endif()

################################################################################
# HPX_PREFIX
################################################################################
# If you want to run hpx applications without installing them, set this to
# ${CMAKE_BINARY_DIR} when using Makefiles or
# ${CMAKE_BINARY_DIR}/$(Configuration) when using Visual Studio
if(HPX_NO_INSTALL)
  if(MSVC)
      set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/$(Configuration)" CACHE PATH "Prefix prepended to install directories." FORCE)
      set(HPX_PREFIX "${CMAKE_BINARY_DIR}/$(Configuration)" CACHE PATH "Where the hpx applications look for component dlls" FORCE)
  else()
      set(HPX_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "Where the hpx applications look for component dlls" FORCE)
  endif()
else()
  set(HPX_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Where the hpx applications look for component dlls" FORCE)
endif()

add_definitions("-DHPX_PREFIX=\"${HPX_PREFIX}\"")

# Make sure the specified build type is valid.
if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR
        "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo" OR
        "${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel" OR
        "${CMAKE_BUILD_TYPE}" STREQUAL "Release"))
  hpx_error("build_type" "\"${CMAKE_BUILD_TYPE}\" is not a valid build type, must be Debug, Release, RelWithDebInfo, or MinSizeRel.")
endif()

# TODO: Move this into a CMake macro.
if(MSVC)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/lib/hpx)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/lib/hpx)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/lib/hpx)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/lib/hpx)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/MinSizeRel/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/MinSizeRel/lib/hpx)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/MinSizeRel/lib/hpx)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo/lib/hpx)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo/lib/hpx)
else()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/hpx)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/hpx)
endif()

################################################################################
# RPATH configuration
################################################################################
if(NOT MSVC)
  set(HPX_RPATH "${CMAKE_INSTALL_PREFIX}/lib/hpx:${CMAKE_BINARY_DIR}/lib/hpx"
      CACHE STRING "Base RPATH for linking" FORCE)
endif()

################################################################################
# C++11 configuration
################################################################################
hpx_option(HPX_CXX11 BOOL "Use C++11 support, if available (default: ON)" ON)

if(HPX_CXX11)
  add_definitions(-DHPX_HAVE_CXX11)
endif()

################################################################################
# SDF output
################################################################################
find_package(HPX_RNPL)

################################################################################
# LORENE
################################################################################
find_package(HPX_LORENE)

################################################################################
# GSL
################################################################################
find_package(HPX_GSL)

################################################################################
# LAPACK
################################################################################
find_package(HPX_LAPACK)

################################################################################
# BLAS
################################################################################
find_package(HPX_BLAS)

################################################################################
# HDF5 output
################################################################################
find_package(HPX_HDF5)
# verify thread safety
hpx_check_for_thread_safe_hdf5(HDF5_THREAD_SAFE)

################################################################################
# EXODUS libraries
################################################################################
find_package(HPX_EXODUS)

################################################################################
# NETCDF
################################################################################
find_package(HPX_NETCDF)

################################################################################
# PAPI event counters
################################################################################
find_package(HPX_PAPI)

################################################################################
# hwloc
################################################################################
find_package(HPX_Hwloc)
if(HWLOC_FOUND)
  hpx_include_sys_directories(${HWLOC_INCLUDE_DIR})
  hpx_link_sys_directories(${HWLOC_LIBRARY_DIR})
  add_definitions(-DHPX_HAVE_HWLOC)
  set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} ${HWLOC_LIBRARY})
endif()

################################################################################
# Qthreads (for benchmark comparisons)
################################################################################
find_package(HPX_Qthreads)

################################################################################
# TBB (for benchmark comparisons)
################################################################################
find_package(HPX_TBB)

################################################################################
# ETI SWARM (for benchmark comparisons)
################################################################################
find_package(HPX_SWARM)

################################################################################
# OpenMP (for benchmark comparisons)
################################################################################
find_package(HPX_OpenMP)

################################################################################
# MPI (for benchmark comparisons)
################################################################################
find_package(HPX_MPI)

################################################################################
# Documentation toolchain (DocBook, BoostBook, QuickBook, xsltproc)
################################################################################
hpx_include(Documentation)

################################################################################
# Warning configuration
################################################################################
hpx_option(HPX_WARNINGS BOOL "Enable compiler warnings (default: ON)" ON ADVANCED)

################################################################################
# Backtrace configuration
################################################################################
hpx_option(HPX_STACKTRACES BOOL "Attach backtraces to HPX exceptions (default: ON)" ON ADVANCED)

if(HPX_STACKTRACES)
  hpx_info("stacktraces" "Stack traces are enabled.")
  add_definitions(-DHPX_HAVE_STACKTRACES)
endif()

################################################################################
# Emulation of SwapContext on Windows
################################################################################
if(MSVC)
  hpx_option(HPX_EMULATE_SWAP_CONTEXT BOOL "Emulate SwapContext API for coroutines (default: OFF)" OFF ADVANCED)

  if(HPX_EMULATE_SWAP_CONTEXT)
    enable_language(ASM_MASM)
    if(CMAKE_ASM_MASM_COMPILER)
      hpx_info("swap_context" "SwitchToFiber emulation is enabled, using compiler: '${CMAKE_ASM_MASM_COMPILER}'")
    else()
      hpx_warning("swap_context" "SwitchToFiber emulation will be disabled, try setting the ASM_MASM environment variable to the assembler executable (ml.exe/ml64.exe)")
      set(HPX_EMULATE_SWAP_CONTEXT OFF CACHE BOOL "Emulate SwitchToFiber API for coroutines (default: OFF)" FORCE)
    endif()
  endif()
endif()

################################################################################
# Native TLS configuration
################################################################################
hpx_option(HPX_NATIVE_TLS BOOL "Use native TLS support if available (default: ON)" ON ADVANCED)

if(HPX_NATIVE_TLS)
  hpx_info("tls" "Native TLS is enabled.")
  add_definitions(-DHPX_HAVE_NATIVE_TLS)
endif()

################################################################################
# pxthread stack size
################################################################################
if(NOT HPX_STACK_SIZE)
  if(MSVC)
    set(HPX_STACK_SIZE "0x4000" CACHE STRING "HPX-thread stack size" FORCE)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR
     "CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    set(HPX_STACK_SIZE "0x10000" CACHE STRING "HPX-thread stack size" FORCE)
  else()
    set(HPX_STACK_SIZE "0x10000" CACHE STRING "HPX-thread stack size" FORCE)
  endif()
else()
  set(HPX_STACK_SIZE "${HPX_STACK_SIZE}" CACHE STRING "HPX-thread stack size" FORCE)
endif()
mark_as_advanced(HPX_STACK_SIZE)

add_definitions(-DHPX_DEFAULT_STACK_SIZE=${HPX_STACK_SIZE})

################################################################################
# Compiler configuration
################################################################################
# Clear CMake defaults
foreach(language CXX Fortran)
  set(CMAKE_${language}_FLAGS_DEBUG "" CACHE STRING "Debug flags (${language})" FORCE)
  set(CMAKE_${language}_FLAGS_MINSIZEREL "" CACHE STRING "MinSizeRel flags (${language})" FORCE)
  set(CMAKE_${language}_FLAGS_RELEASE "" CACHE STRING "Release flags (${language})" FORCE)
  set(CMAKE_${language}_FLAGS_RELWITHDEBINFO ""  CACHE STRING "RelWithDebInfo flags (${language})" FORCE)
endforeach()

if(MSVC)
  add_definitions(-DHPX_BUILD_TYPE="$(Configuration)")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG -DBOOST_DISABLE_ASSERTS")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -DNDEBUG -DBOOST_DISABLE_ASSERTS")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -DDBOOST_DISABLE_ASSERTS")
else()
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DDEBUG)
    add_definitions(-DHPX_BUILD_TYPE="debug")
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    add_definitions(-DNDEBUG)
    add_definitions(-DBOOST_DISABLE_ASSERTS)
    add_definitions(-DHPX_BUILD_TYPE="relwithdebinfo")
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    add_definitions(-DNDEBUG)
    add_definitions(-DBOOST_DISABLE_ASSERTS)
    add_definitions(-DHPX_BUILD_TYPE="minsizerel")
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    add_definitions(-DNDEBUG)
    add_definitions(-DBOOST_DISABLE_ASSERTS)
    add_definitions(-DHPX_BUILD_TYPE="release")
  endif()
endif()

# Debug library postfix
set(CMAKE_DEBUG_POSTFIX "d")

################################################################################
# Utility configuration
################################################################################
hpx_option(HPX_UTIL_BIND BOOL "Use hpx::util::bind instead of boost::bind or std::bind (default: ON)" ON ADVANCED)

if(HPX_UTIL_BIND)
  add_definitions(-DHPX_UTIL_BIND)
endif()

hpx_option(HPX_UTIL_FUNCTION BOOL "Use hpx::util::function instead of boost::function or std::function (default: ON)" ON ADVANCED)

if(HPX_UTIL_FUNCTION)
  add_definitions(-DHPX_UTIL_FUNCTION)
endif()

################################################################################
# Scheduler configuration
################################################################################
hpx_option(HPX_GLOBAL_SCHEDULER BOOL "Enable the use of --queueing=global (default: OFF)" OFF ADVANCED)

if(HPX_GLOBAL_SCHEDULER)
  add_definitions(-DHPX_GLOBAL_SCHEDULER)
endif()

hpx_option(HPX_LOCAL_SCHEDULER BOOL "Enable the use of --queueing=local (default: OFF)" OFF ADVANCED)

if(HPX_LOCAL_SCHEDULER)
  add_definitions(-DHPX_LOCAL_SCHEDULER)
endif()

hpx_option(HPX_ABP_SCHEDULER BOOL "Enable the use of --queueing=abp (default: OFF)" OFF ADVANCED)

if(HPX_ABP_SCHEDULER)
  add_definitions(-DHPX_ABP_SCHEDULER)
endif()

hpx_option(HPX_ABP_PRIORITY_SCHEDULER BOOL "Enable the use of --queueing=priority_abp (default: OFF)" OFF ADVANCED)

if(HPX_ABP_PRIORITY_SCHEDULER)
  add_definitions(-DHPX_ABP_PRIORITY_SCHEDULER)
endif()

hpx_option(HPX_HIERARCHY_SCHEDULER BOOL "Enable the use of --queueing=hierarchy (default: OFF)" OFF ADVANCED)

if(HPX_HIERARCHY_SCHEDULER)
  add_definitions(-DHPX_HIERARCHY_SCHEDULER)
endif()

hpx_option(HPX_PERIODIC_PRIORITY_SCHEDULER BOOL "Enable the use of --queueing=periodic (default: OFF)" OFF ADVANCED)

if(HPX_PERIODIC_PRIORITY_SCHEDULER)
  add_definitions(-DHPX_PERIODIC_PRIORITY_SCHEDULER)
endif()

################################################################################
# MSVC configuration
################################################################################
if(MSVC)
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -Od -Zi -Oy- -MDd -RTC1")
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
        "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -Zi -MD")
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    set(CMAKE_CXX_FLAGS_MINSIZEREL
        "${CMAKE_CXX_FLAGS_MINSIZEREL} -O1 -MD")
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE} -Ox -MD -GL")
  endif()

  # Exceptions
  hpx_append_flag(-EHsc)

  # Runtime type information
  hpx_append_flag(-GR)

  # Multiproccessor build
  hpx_append_flag(-MP)

  # Increase the maximum size of object file sections
  hpx_append_flag(-bigobj LANGUAGES CXX Fortran)

  set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} psapi shlwapi)

  ##############################################################################
  # Stacktraces
  ##############################################################################
  if(HPX_STACKTRACES)
    set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} dbghelp)
  endif()

  ##############################################################################
  # C++11
  ##############################################################################
  if(HPX_CXX11)
    # C++11 support
    hpx_option(HPX_CXX11_RVALUE_REFERENCES BOOL "Compiler supports C++11 rvalue references (default: ON)." ON ADVANCED)
    if(HPX_CXX11_RVALUE_REFERENCES)
      add_definitions(-DHPX_HAVE_CXX11_RVALUE_REFERENCES)
    endif()

    hpx_option(HPX_CXX11_LAMBDAS BOOL "Compiler supports C++11 lambda functions (default: ON)." ON ADVANCED)
    if(HPX_CXX11_LAMBDAS)
      add_definitions(-DHPX_HAVE_CXX11_LAMBDAS)
    endif()

    hpx_option(HPX_CXX11_AUTO BOOL "Compiler supports C++11 auto keyword (default: ON)." ON ADVANCED)
    if(HPX_CXX11_AUTO)
      add_definitions(-DHPX_HAVE_CXX11_AUTO)
    endif()

    hpx_option(HPX_CXX11_UNIQUE_PTR BOOL "Compiler supports C++11 std::unique_ptr (default: ON)." ON ADVANCED)
    if(HPX_CXX11_UNIQUE_PTR)
      add_definitions(-DHPX_HAVE_CXX11_STD_UNIQUE_PTR)
      add_definitions(-DBOOST_LOCKFREE_HAVE_CXX11_UNIQUE_PTR)
    endif()

    hpx_option(HPX_HAVE_CXX11_STD_TUPLE BOOL "Compiler supports C++11 (movable) std::tuple (default: ON)." ON ADVANCED)
    if(HPX_HAVE_CXX11_STD_TUPLE)
      add_definitions(-DHPX_HAVE_CXX11_STD_TUPLE)
    endif()

    hpx_option(HPX_HAVE_CXX11_STD_BIND  BOOL "Compiler supports C++11 (movable) std::bind (default: OFF)." OFF ADVANCED)
    if(NOT HPX_UTIL_BIND)
      add_definitions(-DHPX_HAVE_CXX11_STD_BIND)
    else()
      set(HPX_HAVE_CXX11_STD_BIND OFF CACHE BOOL "Compiler supports C++11 (movable) std::bind." FORCE)
      hpx_info("C++11" "HPX_UTIL_BIND is enabled: even while std::bind is available it will not be used.")
    endif()

    hpx_option(HPX_HAVE_CXX11_STD_FUNCTION BOOL "Compiler supports C++11 (movable) std::function (default: OFF)." OFF ADVANCED)
    if(NOT HPX_UTIL_FUNCTION)
      add_definitions(-DHPX_HAVE_CXX11_STD_FUNCTION)
    else()
      set(HPX_HAVE_CXX11_STD_FUNCTION OFF CACHE BOOL "Compiler supports C++11 (movable) std::function." FORCE)
      hpx_info("C++11" "HPX_UTIL_FUNCTION is enabled: even while std::function is available it will not be used.")
    endif()
  endif()

  ##############################################################################
  # Diagnostics
  ##############################################################################
  if(HPX_WARNINGS)
    hpx_append_flag(-W3)

    # According to the ifort Windows manual, W3 isn't supported
    hpx_append_flag(-W1 LANGUAGES Fortran)
  endif()

  # Display full paths in diagnostics
  hpx_append_flag(-FC)

  ##############################################################################
  # Macro definitions for system headers
  ##############################################################################
  add_definitions(-D_WINDOWS)
  add_definitions(-D_WIN32)
  add_definitions(-D_WIN32_WINNT=0x0601)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)

  ##############################################################################
  # Boost
  ##############################################################################
  # We auto-link Boost on Windows, so don't specify boost libraries explicitly
  set(BOOST_FOUND_LIBRARIES "")
  set(BOOST_serialization_LIBRARY "")

  add_definitions(-DBOOST_USE_WINDOWS_H)
  add_definitions(-DBOOST_SERIALIZATION_DYN_LINK)
  add_definitions(-DBOOST_COROUTINE_USE_FIBERS)
  add_definitions(-DPSAPI_VERSION=1)

################################################################################
# GCC-compatible compiler configuration
################################################################################
else()
  # Workaround for Ubuntu GCC 4.6.2 and Boost 1.49.0 with Debian GCC 4.6.2
  hpx_use_flag_if_available(-Wl,--copy-dt-needed-entries LANGUAGES CXX Fortran
                            NAME copy_dt_needed_entries)

  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    # No optimizations
    hpx_use_flag_if_available(-O0 LANGUAGES CXX Fortran)

    # Generate debugging information
    hpx_use_flag_if_available(-g LANGUAGES CXX Fortran)

    # Keep the frame pointer
    hpx_use_flag_if_available(-fno-omit-frame-pointer LANGUAGES CXX Fortran)

    # Don't inline functions
    hpx_use_flag_if_available(-fno-inline LANGUAGES CXX Fortran)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    # Optimize for speed
    hpx_use_flag_if_available(-O3 LANGUAGES CXX Fortran)

    # Generate debugging information
    hpx_use_flag_if_available(-g LANGUAGES CXX Fortran)

    # Don't inline functions
    hpx_use_flag_if_available(-fno-inline LANGUAGES CXX Fortran)

    # No tail call optimizations
    hpx_use_flag_if_available(-fno-optimize-sibling-calls LANGUAGES CXX Fortran)

    # Don't allocate registers for wide types (e.g. 128bit ints) independently
    hpx_use_flag_if_available(-fno-split-wide-types LANGUAGES CXX Fortran)

    # Disable register allocation webs, which make debugging impossible
    hpx_use_flag_if_available(-fno-web LANGUAGES CXX Fortran)

    # Disable register renaming
    hpx_use_flag_if_available(-fno-rename-registers LANGUAGES CXX Fortran)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    # Optimize for size
    hpx_use_flag_if_available(-Os LANGUAGES CXX Fortran)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    # Optimize for speed
    hpx_use_flag_if_available(-O3 LANGUAGES CXX Fortran)
  endif()

  ##############################################################################
  # Stacktraces
  ##############################################################################
  if(HPX_STACKTRACES)
    # Export all visible symbols, not just those that are used
    hpx_use_flag_if_available(-rdynamic LANGUAGES CXX Fortran)
  endif()

  ##############################################################################
  # C++11
  ##############################################################################
  if(HPX_CXX11)
    # Turn on C++11 support
    hpx_use_flag_if_available(-std=c++0x)

    hpx_check_for_cxx11_rvalue_references(HPX_CXX11_RVALUE_REFERENCES
      DEFINITIONS HPX_HAVE_CXX11_RVALUE_REFERENCES)

    hpx_check_for_cxx11_variadic_templates(HPX_CXX11_VARIADIC_TEMPLATES
      DEFINITIONS HPX_HAVE_CXX11_VARIADIC_TEMPLATES)

    hpx_check_for_cxx11_lambdas(HPX_CXX11_LAMBDAS
      DEFINITIONS HPX_HAVE_CXX11_LAMBDAS)

    hpx_check_for_cxx11_auto(HPX_CXX11_AUTO DEFINITIONS HPX_HAVE_CXX11_AUTO)

    hpx_check_for_cxx11_std_unique_ptr(HPX_CXX11_STD_UNIQUE_PTR
      DEFINITIONS HPX_HAVE_CXX11_STD_UNIQUE_PTR
                  BOOST_LOCKFREE_HAVE_CXX11_STD_UNIQUE_PTR)

    hpx_check_for_cxx11_std_tuple(HPX_CXX11_STD_TUPLE
      DEFINITIONS HPX_HAVE_CXX11_STD_TUPLE)

    if(NOT HPX_UTIL_BIND)
      hpx_check_for_cxx11_std_bind(HPX_CXX11_STD_BIND
        DEFINITIONS HPX_HAVE_CXX11_STD_BIND)
    else()
      hpx_info("C++11" "HPX_UTIL_BIND is enabled: even if std::bind is available it will not be used.")
    endif()

    if(NOT HPX_UTIL_FUNCTION)
      hpx_check_for_cxx11_std_function(HPX_CXX11_STD_FUNCTION
        DEFINITIONS HPX_HAVE_CXX11_STD_FUNCTION)
    else()
      hpx_info("C++11" "HPX_UTIL_FUNCTION is enabled: even if std::function is available it will not be used.")
    endif()
  endif()

  ##############################################################################
  # Diagnostics
  ##############################################################################
  if(HPX_WARNINGS)
    hpx_use_flag_if_available(-Wall LANGUAGES CXX Fortran)
    hpx_use_flag_if_available(-Wno-strict-aliasing LANGUAGES CXX Fortran)
    hpx_use_flag_if_available(-Wsign-promo)
  endif()

  # VLAs are a GNU extensions that we forbid as they are not supported on MSVC
  hpx_use_flag_if_available(-Werror=vla)

  # No return statement in a non-void function can lead to garbage return values
  # in GCC.
  hpx_use_flag_if_available(-Werror=return-type)

  # Show the flags that toggle each warning
  hpx_use_flag_if_available(-fdiagnostics-show-option LANGUAGES CXX Fortran)

  # We get false positives all over the place with this. Also, detection for
  # this flag fails with GCC 4.4 and 4.5.
  if(040600 LESS ${GCC_VERSION} OR 040600 EQUAL ${GCC_VERSION})
    hpx_use_flag_if_available(-Wno-unused-but-set-parameter)
    hpx_use_flag_if_available(-Wno-unused-but-set-variable)

    # Uninitialized variables are bad, earlier compilers issue spurious warnings
    hpx_use_flag_if_available(-Werror=uninitialized)
  endif()

  # Silence warning about __sync_fetch_and_nand changing semantics
  hpx_use_flag_if_available(-Wno-sync-nand)

  ##############################################################################
  # GCC-only configuration
  ##############################################################################
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if(040400 LESS ${GCC_VERSION})
      option(HPX_ELF_HIDDEN_VISIBILITY
        "Use -fvisibility=hidden for Release and MinSizeRel builds (default: ON)" ON)

      if(HPX_ELF_HIDDEN_VISIBILITY)
        if("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel" OR
       "${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR
       "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
          add_definitions(-DHPX_ELF_HIDDEN_VISIBILITY)
          add_definitions(-DBOOST_COROUTINE_GCC_HAVE_VISIBILITY)
          add_definitions(-DBOOST_PLUGIN_GCC_HAVE_VISIBILITY)
          hpx_use_flag_if_available(-fvisibility=hidden LANGUAGES CXX Fortran)
        endif()
      endif()
    endif()
  endif()

  ##############################################################################
  # x86-64 specific configuration
  ##############################################################################
  # GCC's -march=native will automatically tune generated code for the host
  # environment. This is available on newish versions of GCC only (4.3ish). If
  # this flag is used, the generated binaries will be less portable.
  hpx_use_flag_if_available(-march=native LANGUAGES CXX Fortran)

  # cmpxchg16b is an x86-64 extension present on most newer x86-64 machines.
  # It allows us to do double quadword (128bit) atomic compare and swap
  # operations, which is AWESOME. Note that early x86-64 processors do lack
  # this instruction.
  hpx_use_flag_if_available(-mcx16 LANGUAGES CXX Fortran)

  if(HPX_CXX_FLAG_MCX16)
    add_definitions(-DHPX_HAVE_GNU_SYNC_16)
    add_definitions(-DBOOST_ATOMIC_HAVE_GNU_SYNC_16) # for the gnu code
  endif()

  # __attribute__ ((aligned(16))) should align a variable to a 16-byte, however,
  # GCC sets an upper limit on the maximum alignment (__BIGGEST_ALIGNMENT__)
  # and some versions don't warn if you ask for an alignment above said limit.
  # Instead, they'll just silently use the maximum, which can be problematical.
  hpx_check_for_gnu_aligned_16(HPX_GNU_ALIGNED_16
    DEFINITIONS HPX_HAVE_GNU_ALIGNED_16
                BOOST_ATOMIC_HAVE_GNU_ALIGNED_16) # for the gnu code

  # __uint128_t and __int128_t are a nifty, albeit undocumented, GNU extension
  # that's been supported in GCC (4.1ish and up) and clang-linux for awhile
  # (strangely, intel-linux doesn't support this). This is particularly useful
  # for use with cmpxchg16b
  hpx_check_for_gnu_128bit_integers(HPX_GNU_128BIT_INTEGERS
    DEFINITIONS HPX_HAVE_GNU_128BIT_INTEGERS
                BOOST_ATOMIC_HAVE_GNU_128BIT_INTEGERS) # for integral casts

  # we use movdqa for atomic 128bit loads and stores
  hpx_cpuid("sse2" HPX_SSE2
    DEFINITIONS HPX_HAVE_SSE2
                BOOST_ATOMIC_HAVE_SSE2)

  if(HPX_SSE2)
    hpx_use_flag_if_available(-msse2 LANGUAGES CXX Fortran)
  endif()

  # rdtsc is an x86 instruction that reads the value of a CPU time stamp
  # counter. rdtscp is an extension to rdtsc. The difference is that rdtscp is
  # a serializing instruction.
  hpx_cpuid("rdtsc" HPX_RDTSC
    DEFINITIONS HPX_HAVE_RDTSC)
  hpx_cpuid("rdtscp" HPX_RDTSCP
    DEFINITIONS HPX_HAVE_RDTSCP)

  ##############################################################################
  # Macro definitions for system headers
  ##############################################################################
  add_definitions(-D_GNU_SOURCE)

  ##############################################################################
  # System libraries
  ##############################################################################
  set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} dl rt)

  hpx_check_for_pthread_affinity_np(HPX_PTHREAD_AFFINITY_NP
    DEFINITIONS HPX_HAVE_PTHREAD_AFFINITY_NP)

  hpx_use_flag_if_available(-pthread)

  set(HPX_GOOGLE_PERFTOOLS ON CACHE BOOL "Compile HPX with google perftools" FORCE)

  if(HPX_GOOGLE_PERFTOOLS)
    find_package(HPX_GooglePerftools)
    if(GOOGLE_PERFTOOLS_FOUND AND HPX_GOOGLE_PERFTOOLS)
      set(hpx_LIBRARIES ${hpx_LIBRARIES} ${GOOGLE_PERFTOOLS_LIBRARY})
      hpx_include_sys_directories(${GOOGLE_PERFTOOLS_INCLUDE_DIR})
      hpx_link_sys_directories(${GOOGLE_PERFTOOLS_LIBRARY_DIR})
      add_definitions(-DHPX_GOOGLE_PERFTOOLS)
    else()
      set(GOOGLE_PERFTOOLS_LIBRARY "")
    endif()
  else()
    set(GOOGLE_PERFTOOLS_LIBRARY "")
  endif()

  if(NOT HPX_MALLOC)
    set(HPX_MALLOC "TCMalloc")
  endif()

  find_package(HPX_TCMalloc)
  find_package(HPX_Jemalloc)

  if("${HPX_MALLOC}" MATCHES "tcmalloc|TCMalloc|TCMALLOC" AND NOT TCMALLOC_FOUND)
    hpx_warn("malloc" "tcmalloc allocator not found.")
  endif()

  if("${HPX_MALLOC}" MATCHES "jemalloc|Jemalloc|JEMALLOC" AND NOT JEMALLOC_FOUND)
    hpx_warn("malloc" "jemalloc allocator not found.")
  endif()

  set(hpx_MALLOC_LIBRARY "")

  if(NOT "${HPX_MALLOC}" MATCHES "system|System|SYSTEM")
    if(TCMALLOC_FOUND OR JEMALLOC_FOUND)
      if("${HPX_MALLOC}" MATCHES "tcmalloc|TCMalloc|TCMALLOC" OR NOT JEMALLOC_FOUND)
        hpx_info("malloc" "Using tcmalloc allocator.")
        set(hpx_MALLOC_LIBRARY ${TCMALLOC_LIBRARY})
        set(hpx_LIBRARIES ${hpx_LIBRARIES} ${TCMALLOC_LIBRARY})
        hpx_include_sys_directories(${TCMALLOC_INCLUDE_DIR})
        hpx_link_sys_directories(${TCMALLOC_LIBRARY_DIR})
        add_definitions(-DHPX_TCMALLOC)
      else()
        hpx_info("malloc" "Using jemalloc allocator.")
        set(hpx_MALLOC_LIBRARY ${JEMALLOC_LIBRARY})
        set(hpx_LIBRARIES ${hpx_LIBRARIES} ${JEMALLOC_LIBRARY})
        hpx_include_sys_directories(${JEMALLOC_INCLUDE_DIR})
        hpx_link_sys_directories(${JEMALLOC_LIBRARY_DIR})
        add_definitions(-DHPX_JEMALLOC)
      endif()

      hpx_use_flag_if_available(-fno-builtin-cfree)
      hpx_use_flag_if_available(-fno-builtin-pvalloc)
      hpx_use_flag_if_available(-fno-builtin-malloc)
      hpx_use_flag_if_available(-fno-builtin-free)
      hpx_use_flag_if_available(-fno-builtin-calloc)
      hpx_use_flag_if_available(-fno-builtin-realloc)
      hpx_use_flag_if_available(-fno-builtin-valloc)
      hpx_use_flag_if_available(-fno-builtin-memalign)
      hpx_use_flag_if_available(-fno-builtin-posix_memalign)
    else()
      hpx_info("malloc" "Using system allocator.")
      hpx_warn("malloc"
        "HPX will perform poorly without tcmalloc or jemalloc. See docs/refmanual/source/build/linux/malloc_allocators.rst.")
    endif()
  else()
    hpx_info("malloc" "Using system allocator.")
    hpx_warn("malloc"
      "HPX will perform poorly without tcmalloc or jemalloc. See docs/refmanual/source/build/linux/malloc_allocators.rst.")
  endif()
endif()

################################################################################
# target specification
################################################################################
# hpx_LIBRARIES lists all libraries a HPX module needs to be linked with by
# default.
set(hpx_LIBRARIES hpx hpx_serialization)

# Recurse into some subdirectories. This does not actually cause another cmake
# executable to run. The same process will walk through the project's entire
# directory structure.
add_subdirectory(src)

hpx_option(HPX_BUILD_EXAMPLES BOOL "Build HPX examples (default: ON)" ON ADVANCED)

if(HPX_BUILD_EXAMPLES)
  add_hpx_pseudo_target(examples)
  hpx_include_directories(examples)
  add_subdirectory(examples)
endif()

hpx_option(HPX_BUILD_TESTS BOOL "Build HPX tests (default: ON)" ON ADVANCED)

if(HPX_BUILD_TESTS)
  hpx_option(HPX_BUILD_TESTS_BENCHMARKS BOOL "Build HPX benchmark tests (default: ON)" ON ADVANCED)
  hpx_option(HPX_BUILD_TESTS_REGRESSIONS BOOL "Build HPX regression tests (default: ON)" ON ADVANCED)
  hpx_option(HPX_BUILD_TESTS_UNIT BOOL "Build HPX unit tests (default: ON)" ON ADVANCED)

  add_hpx_pseudo_target(tests)
  hpx_include_directories(tests)
  add_subdirectory(tests)
else()
  unset(HPX_BUILD_TESTS_BENCHMARKS CACHE)
  unset(HPX_BUILD_TESTS_REGRESSIONS CACHE)
  unset(HPX_BUILD_TESTS_UNIT CACHE)
endif()

hpx_option(HPX_BUILD_RUNTIME BOOL "Build HPX runtime (default: ON)" ON ADVANCED)

if(HPX_BUILD_RUNTIME)
  add_hpx_pseudo_target(runtime)
  add_subdirectory(runtime)
endif()

hpx_option(HPX_BUILD_TOOLS BOOL "Build HPX tools (default: OFF)" OFF ADVANCED)

if(HPX_BUILD_TOOLS)
  add_hpx_pseudo_target(tools)
  add_subdirectory(tools)
endif()

if(HPX_BUILD_DOCUMENTATION)
  add_subdirectory(docs)
endif()

################################################################################
# installation instructions
################################################################################
if(NOT HPX_NO_INSTALL)
  install( # install all hpx header files
    DIRECTORY hpx/
    DESTINATION include/hpx
    COMPONENT core
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE
    PATTERN ".git" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "CTestFiles" EXCLUDE)

  install( # install all hpx cmake utility files
    DIRECTORY cmake/
    DESTINATION share/hpx-${HPX_VERSION}/cmake
    COMPONENT core
    PATTERN ".cmake.in" EXCLUDE
    PATTERN ".svn" EXCLUDE
    PATTERN ".git" EXCLUDE)

  install( # install hpx python scripts
    DIRECTORY python/scripts/
    DESTINATION bin
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                     GROUP_READ GROUP_EXECUTE
                     WORLD_READ WORLD_EXECUTE
    COMPONENT core
    FILES_MATCHING PATTERN "*.py"
    PATTERN ".svn" EXCLUDE
    PATTERN ".git" EXCLUDE)

  if(UNIX)
    file(GLOB scripts "${CMAKE_SOURCE_DIR}/python/scripts/*.py")
    foreach(script ${scripts})
      get_filename_component(script_name ${script} NAME)
      get_filename_component(script_name_we ${script} NAME_WE)
      install(CODE
        "set(bindir \"${CMAKE_INSTALL_PREFIX}/bin/\")
         execute_process(
           COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
                   \"${script_name}\" \"${script_name_we}\"
           WORKING_DIRECTORY \"\${bindir}\")")
    endforeach()
  endif()

  install( # install hpx python module (TODO: this is a temporary hack)
    DIRECTORY python/hpx
    DESTINATION share/hpx-${HPX_VERSION}/python
    COMPONENT core
    FILES_MATCHING PATTERN "*.py"
    PATTERN ".svn" EXCLUDE
    PATTERN ".git" EXCLUDE)

  install( # install external dependencies
    DIRECTORY external/atomic/boost
              external/cache/boost
              external/coroutine/boost
              external/endian/boost
              external/lockfree/boost
              external/logging/boost
              external/plugin/boost
              external/serialization/boost
    DESTINATION include
    COMPONENT core
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE
    PATTERN ".git" EXCLUDE)

  if(HPX_INTERNAL_MOVE)
    install( # install external dependencies
      DIRECTORY external/move/boost
      DESTINATION include
      COMPONENT core
      FILES_MATCHING PATTERN "*.hpp"
      PATTERN ".svn" EXCLUDE
      PATTERN ".git" EXCLUDE)
  endif()

  if(HPX_INTERNAL_CHRONO)
    install( # install Boost.Chrono headers if we're using our internal version
      DIRECTORY external/chrono/boost
      DESTINATION include
      COMPONENT core
      FILES_MATCHING PATTERN "*.hpp"
      PATTERN ".svn" EXCLUDE
      PATTERN ".git" EXCLUDE)
  endif()

  install(
    FILES ${hpx_SOURCE_DIR}/LICENSE_1_0.txt
    DESTINATION share/hpx-${HPX_VERSION}
    COMPONENT license)

  if(HPX_BUILD_DOCUMENTATION)
    # install hpx documentation files
    if(MSVC)
      set(doc_dir ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
    else()
      set(doc_dir ${CMAKE_BINARY_DIR})
    endif()

    install(
      DIRECTORY ${doc_dir}/share/hpx-${HPX_VERSION}/docs/
      DESTINATION share/hpx-${HPX_VERSION}/doc
      COMPONENT docs
      OPTIONAL
      FILES_MATCHING PATTERN "*.html"
      PATTERN ".svn" EXCLUDE
      PATTERN ".git" EXCLUDE)

    install(
      FILES ${hpx_SOURCE_DIR}/docs/index.html
      DESTINATION share/hpx-${HPX_VERSION}/
      COMPONENT docs)

    install(
      DIRECTORY ${hpx_SOURCE_DIR}/docs/html/src/
      DESTINATION share/hpx-${HPX_VERSION}/doc/html/src
      COMPONENT docs
      FILES_MATCHING PATTERN "*.css"
      PATTERN ".svn" EXCLUDE
      PATTERN ".git" EXCLUDE)

    install(
      DIRECTORY ${hpx_SOURCE_DIR}/docs/html/images/
      DESTINATION share/hpx-${HPX_VERSION}/doc/html/images
      COMPONENT docs
      FILES_MATCHING PATTERN "*.png"
      PATTERN ".svn" EXCLUDE
      PATTERN ".git" EXCLUDE)
  endif()
endif()

################################################################################
# External build system support (FindHPX.cmake and pkg-config).
################################################################################
set(HPX_LINK_DIRECTORIES
    ${CMAKE_INSTALL_PREFIX}/lib/hpx ${HPX_LINK_DIRECTORIES})
set(HPX_INCLUDE_DIRECTORIES
    ${CMAKE_INSTALL_PREFIX}/include ${HPX_INCLUDE_DIRECTORIES})

list(REMOVE_DUPLICATES HPX_LINK_DIRECTORIES)
list(REMOVE_DUPLICATES HPX_INCLUDE_DIRECTORIES)

set(external_rpath "")
set(external_link_directories "")
set(external_link_flags "")
set(external_include_directories "")
set(external_include_flags "")

foreach(directory ${HPX_LINK_DIRECTORIES})
  set(external_link_directories "${external_link_directories} ${directory}")
  set(external_link_flags "${external_link_flags} -L${directory}")
  set(external_rpath "${external_rpath}:${directory}")
endforeach()

foreach(directory ${HPX_INCLUDE_DIRECTORIES})
  set(external_include_directories "${external_include_directories} ${directory}")
  set(external_include_flags "${external_include_flags} -I${directory}")
endforeach()

get_directory_property(external_definitions DEFINITIONS)

set(cmake_dir cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})

if(MSVC)
  set(output_dir ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
else()
  set(output_dir ${CMAKE_BINARY_DIR})
endif()

if(HPX_NO_INSTALL)
  set(external_cmake_path ${hpx_SOURCE_DIR}/cmake)
else()
  set(external_cmake_path ${HPX_PREFIX}/share/hpx-${HPX_VERSION}/cmake)
endif()

configure_file(${hpx_SOURCE_DIR}/cmake/templates/FindHPX.cmake.in
               ${output_dir}/share/${cmake_dir}/Modules/FindHPX.cmake
               ESCAPE_QUOTES @ONLY)

if(NOT HPX_NO_INSTALL)
  install(FILES ${output_dir}/share/${cmake_dir}/Modules/FindHPX.cmake
          DESTINATION share/${cmake_dir}/Modules)
endif()

configure_file(${hpx_SOURCE_DIR}/cmake/templates/hpx_application.pc.in
               ${output_dir}/lib/pkgconfig/hpx_application.pc
               @ONLY)

configure_file(${hpx_SOURCE_DIR}/cmake/templates/hpx_component.pc.in
               ${output_dir}/lib/pkgconfig/hpx_component.pc
               @ONLY)

if(NOT HPX_NO_INSTALL)
  install(FILES ${output_dir}/lib/pkgconfig/hpx_application.pc
          DESTINATION lib/pkgconfig)
  install(FILES ${output_dir}/lib/pkgconfig/hpx_component.pc
          DESTINATION lib/pkgconfig)
endif()

if(DOXYGEN_FOUND)
  # the list of files to pass to doxygen is defined in docs/CMakeLists.txt
  set(doxygen_inputs_list
      "${hpx_SOURCE_DIR}/hpx/hpx_fwd.hpp"
      "${hpx_SOURCE_DIR}/hpx/hpx_init.hpp"
      "${hpx_SOURCE_DIR}/hpx/exception.hpp"
      "${hpx_SOURCE_DIR}/hpx/runtime/actions/action_support.hpp"
      "${hpx_SOURCE_DIR}/hpx/runtime/actions/plain_action.hpp"
      "${hpx_SOURCE_DIR}/hpx/runtime/actions/component_action.hpp")

  foreach(doxygen_input ${doxygen_inputs_list})
    set(doxygen_inputs "${doxygen_inputs} ${doxygen_input}")
  endforeach()
  #hpx_info("doxygen" "${doxygen_inputs}")

  set(doxygen_output_file "${CMAKE_CURRENT_BINARY_DIR}/docs/hpx_autodoc")
  set(doxygen_output_dir "${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen")

  set(doxygen_definition_list
      "DOXYGEN:=1"
      "BOOST_SCOPED_ENUM_START(x):=enum x"
      "BOOST_SCOPED_ENUM_END="
      "HPX_EXCEPTION_EXPORT="
      "HPX_EXPORT="
      "HPX_API_EXPORT="
      "HPX_STD_FUNCTION=std::function"
      "extern=")

  foreach(doxygen_predef ${doxygen_definition_list})
    set(doxygen_definitions "${doxygen_definitions} \"${doxygen_predef}\"")
  endforeach()
  #hpx_info("doxygen" "${doxygen_definitions}")

  hpx_info("doxygen" "Creating ${CMAKE_CURRENT_BINARY_DIR}/docs/hpx_autodoc.doxy")
  configure_file(${hpx_SOURCE_DIR}/cmake/templates/autodoc.doxy.in
                 ${CMAKE_CURRENT_BINARY_DIR}/docs/hpx_autodoc.doxy
                 @ONLY)
endif()

file(MAKE_DIRECTORY ${output_dir}/share/hpx-${HPX_VERSION}/tests)

file(WRITE ${output_dir}/share/hpx-${HPX_VERSION}/tests/hpx_unit.tests
           "[\n${HPX_UNIT_TEST_LIST}\n]\n")

file(WRITE ${output_dir}/share/hpx-${HPX_VERSION}/tests/hpx_regression.tests
           "[\n${HPX_REGRESSION_TEST_LIST}\n]\n")

if(NOT HPX_NO_INSTALL)
  install(FILES ${output_dir}/share/hpx-${HPX_VERSION}/tests/hpx_unit.tests
          DESTINATION share/hpx-${HPX_VERSION}/tests)
  install(FILES ${output_dir}/share/hpx-${HPX_VERSION}/tests/hpx_regression.tests
          DESTINATION share/hpx-${HPX_VERSION}/tests)
endif()

unset(HPX_UNIT_TEST_LIST CACHE)
unset(HPX_REGRESSION_TEST_LIST CACHE)

