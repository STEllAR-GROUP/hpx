# Copyright (c)      2020 Mikael Simberg
# Copyright (c) 2007-2021 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
# Copyright (c)      2013 Jeroen Habraken
# Copyright (c) 2014-2016 Andreas Schaefer
# Copyright (c) 2017      Abhimanyu Rawat
# Copyright (c) 2017      Google
# Copyright (c) 2017      Taeguk Kwon
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Overrides must go before the project() statement, otherwise they are ignored.

# ##############################################################################
# C++ overrides
# ##############################################################################
set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HPX_CXXOverrides.cmake"
)

# ##############################################################################
# Build type (needs to be handled before project command below)
# ##############################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE
        STRING
        "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)"
        FORCE
  )
endif()

# ##############################################################################
# project metadata
# ##############################################################################
project(HPX CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ##############################################################################
# Special handling for building tests/examples against a previously installed
# version of HPX
# ##############################################################################
if(HPX_WITH_INSTALLED_VERSION)
  include(cmake/installed_hpx.cmake)
  return()
endif()

# ##############################################################################
# Build all of HPX
# ##############################################################################
set(HPX_VERSION_MAJOR 1)
set(HPX_VERSION_MINOR 8)
set(HPX_VERSION_SUBMINOR 0)
set(HPX_VERSION_DATE 20210714)
set(HPX_VERSION_TAG "-trunk")

set(HPX_VERSION
    "${HPX_VERSION_MAJOR}.${HPX_VERSION_MINOR}.${HPX_VERSION_SUBMINOR}"
)
set(HPX_LIBRARY_VERSION "${HPX_VERSION}")
set(HPX_SOVERSION ${HPX_VERSION_MAJOR})
set(HPX_PACKAGE_NAME HPX)
# To keep track of the hpx_root when other subprojects are declared
set(HPX_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(HPX_BINARY_DIR "${PROJECT_BINARY_DIR}")

# ##############################################################################
# CMake configuration
# ##############################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(HPX_Utils)

# explicitly set certain policies
cmake_policy(VERSION 3.18)
hpx_set_cmake_policy(CMP0042 NEW)
hpx_set_cmake_policy(CMP0060 NEW)
hpx_set_cmake_policy(CMP0074 NEW)

# We save the passed compiler flag to a special variable. This is needed for our
# build system unit tests. Some flags might influence the created symbols
# (_GLIBCXX_DEBUG i look at you)
set(CMAKE_CXX_FLAGS_SAFE ${CMAKE_CXX_FLAGS})
include(HPX_CheckCXXStandard)

include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

# include additional macro definitions
include(HPX_AddDefinitions)
include(HPX_CreateSymbolicLink)

hpx_force_out_of_tree_build(
  "This project requires an out-of-source-tree build. See README.rst. Clean your CMake cache and CMakeFiles if this message persists."
)

if(NOT HPX_CMAKE_LOGLEVEL)
  set(HPX_CMAKE_LOGLEVEL "WARN")
endif()

# print initial diagnostics
hpx_info("CMake version: ${CMAKE_VERSION}")
hpx_info("HPX version: ${HPX_VERSION}")

# ##############################################################################
# reset cached variables that need to be re-filled
unset(HPX_COMPONENTS CACHE)
unset(HPX_EXPORT_TARGETS CACHE)
unset(HPX_EXPORT_INTERNAL_TARGETS CACHE)
unset(HPX_LIBS CACHE)
unset(HPX_ENABLED_MODULES CACHE)
unset(HPX_STATIC_PARCELPORT_PLUGINS CACHE)

# ##############################################################################
# Set up dummy compiler flags targets
# ##############################################################################
include(HPX_CompilerFlagsTargets)

# ##############################################################################
# Setup platform for which HPX should be compiled for.
#
include(HPX_SetPlatform)
if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
  unset(HPX_LIBRARY_VERSION)
  unset(HPX_SOVERSION)
endif()

if(MSVC)
  hpx_option(
    HPX_WITH_VS_STARTUP_PROJECT STRING
    "Define the startup project for the HPX solution (default: ALL_BUILD)."
    "ALL_BUILD" ADVANCED
  )
  if(HPX_WITH_VS_STARTUP_PROJECT)
    set(VS_STARTUP_PROJECT ${HPX_WITH_VS_STARTUP_PROJECT})
  endif()
endif()

# ##############################################################################

# ##############################################################################
# Set our build options cache variables which are customizable by users
#

hpx_option(
  HPX_WITH_DEPRECATION_WARNINGS BOOL
  "Enable warnings for deprecated facilities. (default: ON)" ON ADVANCED
)

if(HPX_WITH_DEPRECATION_WARNINGS)
  # enable deprecation warnings globally
  hpx_add_config_cond_define(HPX_HAVE_DEPRECATION_WARNINGS 1)
endif()

# Generic build options
hpx_option(
  HPX_WITH_FAULT_TOLERANCE
  BOOL
  "Build HPX to tolerate failures of nodes, i.e. ignore errors in active communication channels (default: OFF)"
  OFF
  ADVANCED
)
if(HPX_WITH_FAULT_TOLERANCE)
  hpx_add_config_define(HPX_HAVE_FAULT_TOLERANCE)
endif()

# Compiler related build options
hpx_option(
  HPX_WITH_GCC_VERSION_CHECK BOOL
  "Don't ignore version reported by gcc (default: ON)" ON ADVANCED
)

hpx_option(
  HPX_WITH_COMPILER_WARNINGS BOOL "Enable compiler warnings (default: ON)" ON
  ADVANCED
)

hpx_option(
  HPX_WITH_COMPILER_WARNINGS_AS_ERRORS BOOL
  "Turn compiler warnings into errors (default: OFF)" OFF ADVANCED
)

hpx_option(
  HPX_WITH_EXECUTABLE_PREFIX STRING
  "Executable prefix (default none), 'hpx_' useful for system install." ""
  CATEGORY "Build Targets"
)

hpx_option(
  HPX_WITH_DOCUMENTATION BOOL "Build the HPX documentation (default OFF)." OFF
  CATEGORY "Build Targets"
)

if(HPX_WITH_DOCUMENTATION)
  set(valid_output_formats html singlehtml latexpdf man)
  hpx_option(
    HPX_WITH_DOCUMENTATION_OUTPUT_FORMATS
    STRING
    "List of documentation output formats to generate. Valid options are ${valid_output_formats}. Multiple values can be separated with semicolons. (default html)."
    "html"
    CATEGORY "Build Targets"
  )

  foreach(output_format ${HPX_WITH_DOCUMENTATION_OUTPUT_FORMATS})
    if(NOT ${output_format} IN_LIST valid_output_formats)
      hpx_error(
        "${output_format} is not a valid value for HPX_WITH_DOCUMENTATION_OUTPUT_FORMATS. Valid output format are: ${valid_output_formats}."
      )
    endif()
  endforeach()
endif()

if(WIN32)
  set(HPX_WITH_PSEUDO_DEPENDENCIES
      OFF
      CACHE INTERNAL "" FORCE
  )
else()
  set(HPX_WITH_PSEUDO_DEPENDENCIES
      ON
      CACHE INTERNAL "" FORCE
  )
endif()

hpx_option(
  HPX_WITH_UNITY_BUILD BOOL
  "Enable unity build for certain build targets (default OFF)" OFF ADVANCED
)
if(HPX_WITH_UNITY_BUILD)
  set(HPX_WITH_UNITY_BUILD_OPTION UNITY_BUILD)
endif()

hpx_option(
  HPX_WITH_PRECOMPILED_HEADERS
  BOOL
  "Enable precompiled headers for certain build targets (experimental) (default OFF)"
  OFF
  ADVANCED
)
if(HPX_WITH_PRECOMPILED_HEADERS)
  set(HPX_WITH_PRECOMPILED_HEADERS_INTERNAL ON)
  # Only create the targets here. They will be set up later once all modules are
  # known.
  add_library(hpx_precompiled_headers OBJECT libs/src/dummy.cpp)
  add_executable(hpx_exe_precompiled_headers libs/src/dummy.cpp)

  set_target_properties(hpx_precompiled_headers PROPERTIES FOLDER "Core")
  set_target_properties(hpx_exe_precompiled_headers PROPERTIES FOLDER "Core")
endif()

# ##############################################################################
# Dynamic hpx_main
# ##############################################################################

set(HPX_WITH_DYNAMIC_HPX_MAIN_DEFAULT OFF)
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" OR APPLE)
  set(HPX_WITH_DYNAMIC_HPX_MAIN_DEFAULT ON)
endif()

hpx_option(
  HPX_WITH_DYNAMIC_HPX_MAIN
  BOOL
  "Enable dynamic overload of system ``main()`` (Linux and Apple only, default: ON)"
  ${HPX_WITH_DYNAMIC_HPX_MAIN_DEFAULT}
  ADVANCED
)
if(HPX_WITH_DYNAMIC_HPX_MAIN)
  if(NOT HPX_WITH_DYNAMIC_HPX_MAIN_DEFAULT)
    hpx_error(
      "HPX_WITH_DYNAMIC_HPX_MAIN was set to ON, but the option is only available on Linux and Apple (this is \"${CMAKE_SYSTEM_NAME}\")."
    )
  endif()
  hpx_add_config_define(HPX_HAVE_DYNAMIC_HPX_MAIN)
endif()

# ##############################################################################
# Some platforms do not support dynamic linking. Enable this to link all
# libraries statically. This also changes some of the internals of HPX related
# to how components are loaded.
# ##############################################################################
hpx_option(
  HPX_WITH_STATIC_LINKING BOOL
  "Compile HPX statically linked libraries (Default: OFF)" OFF ADVANCED
)
if(HPX_WITH_STATIC_LINKING)
  hpx_add_config_define(HPX_HAVE_STATIC_LINKING)
  set(hpx_library_link_mode STATIC)
  set(CMAKE_SHARED_LIBS OFF)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})

  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    hpx_option(
      HPX_WITH_STATIC_EXE_LINKING BOOL
      "Compile HPX statically linked executables (Default: OFF)" OFF ADVANCED
    )

    if(HPX_WITH_STATIC_EXE_LINKING)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
    endif()
  endif()
else()
  set(hpx_library_link_mode SHARED)
endif()

# ##############################################################################

hpx_option(
  HPX_WITH_EXAMPLES BOOL "Build the HPX examples (default ON)" ON
  CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS BOOL "Build the HPX tests (default ON)" ON
  CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_BENCHMARKS BOOL "Build HPX benchmark tests (default: ON)" ON
  ADVANCED CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_REGRESSIONS BOOL "Build HPX regression tests (default: ON)" ON
  ADVANCED CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_UNIT BOOL "Build HPX unit tests (default: ON)" ON ADVANCED
  CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_HEADERS BOOL "Build HPX header tests (default: OFF)" OFF
  ADVANCED CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_EXTERNAL_BUILD BOOL
  "Build external cmake build tests (default: ON)" ON ADVANCED
  CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TESTS_EXAMPLES BOOL "Add HPX examples as tests (default: ON)" ON
  ADVANCED CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_TOOLS BOOL "Build HPX tools (default: OFF)" OFF ADVANCED
  CATEGORY "Build Targets"
)

hpx_option(
  HPX_WITH_COMPILE_ONLY_TESTS BOOL
  "Create build system support for compile time only HPX tests (default ON)" ON
  CATEGORY "Build Targets"
)
hpx_option(
  HPX_WITH_FAIL_COMPILE_TESTS BOOL
  "Create build system support for fail compile HPX tests (default ON)" ON
  CATEGORY "Build Targets"
)

# disable all tests if HPX_WITH_TESTS=OFF
if(NOT HPX_WITH_TESTS)
  hpx_set_option(
    HPX_WITH_TESTS_BENCHMARKS
    VALUE OFF
    FORCE
  )
  hpx_set_option(
    HPX_WITH_TESTS_REGRESSIONS
    VALUE OFF
    FORCE
  )
  hpx_set_option(
    HPX_WITH_TESTS_UNIT
    VALUE OFF
    FORCE
  )
  hpx_set_option(
    HPX_WITH_TESTS_HEADERS
    VALUE OFF
    FORCE
  )
  hpx_set_option(
    HPX_WITH_TESTS_EXTERNAL_BUILD
    VALUE OFF
    FORCE
  )
  hpx_set_option(
    HPX_WITH_TESTS_EXAMPLES
    VALUE OFF
    FORCE
  )
endif()

# Enable IO-counters on linux systems only
set(HPX_WITH_IO_COUNTERS_DEFAULT OFF)
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(HPX_WITH_IO_COUNTERS_DEFAULT ON)
endif()

hpx_option(
  HPX_WITH_IO_COUNTERS BOOL
  "Enable IO counters (default: ${HPX_WITH_IO_COUNTERS_DEFAULT})"
  ${HPX_WITH_IO_COUNTERS_DEFAULT} ADVANCED CATEGORY "Build Targets"
)
if(HPX_WITH_IO_COUNTERS)
  if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    hpx_error(
      "HPX_WITH_IO_COUNTERS was set to ON, but IO counters are only available on Linux (this is \"${CMAKE_SYSTEM_NAME}\")"
    )
  endif()
  hpx_add_config_define(HPX_HAVE_IO_COUNTERS)
endif()

set(HPX_FULL_RPATH_DEFAULT ON)
if(APPLE OR WIN32)
  set(HPX_FULL_RPATH_DEFAULT OFF)
endif()
hpx_option(
  HPX_WITH_FULL_RPATH
  BOOL
  "Build and link HPX libraries and executables with full RPATHs (default: ${HPX_FULL_RPATH_DEFAULT})"
  ${HPX_FULL_RPATH_DEFAULT}
  ADVANCED
)

# ##############################################################################
# pkgconfig file generation
# ##############################################################################
hpx_option(
  HPX_WITH_PKGCONFIG
  BOOL
  "Enable generation of pkgconfig files (default: ON on Linux without CUDA/HIP, otherwise OFF)"
  ON
  ADVANCED
)

# ##############################################################################
# Utility configuration
# ##############################################################################

# We create a target to contain libraries like rt, dl etc. in order to remove
# global variables
add_library(hpx_base_libraries INTERFACE)

# Options for automatically fetching HPXLocal
hpx_option(
  HPX_WITH_FETCH_HPXLOCAL
  BOOL
  "Use FetchContent to fetch HPXLocal. By default an installed HPXLocal will be used. (default: ON)"
  ON
  CATEGORY "Build Targets"
  ADVANCED
)
set(HPX_WITH_HPXLOCAL_TAG_DEFAULT "hpx-local-pre-release-1")
hpx_option(
  HPX_WITH_HPXLOCAL_TAG STRING "HPXLocal repository tag or branch"
  "${HPX_WITH_HPXLOCAL_TAG_DEFAULT}"
  CATEGORY "Build Targets"
  ADVANCED
)

set(HPX_HIDDEN_VISIBILITY_DEFAULT ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
    set(HPX_HIDDEN_VISIBILITY_DEFAULT OFF)
  endif()
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(HPX_HIDDEN_VISIBILITY_DEFAULT OFF)
endif()
if(APPLE)
  set(HPX_HIDDEN_VISIBILITY_DEFAULT OFF)
endif()

hpx_option(
  HPX_WITH_HIDDEN_VISIBILITY
  BOOL
  "Use -fvisibility=hidden for builds on platforms which support it (default: ${HPX_HIDDEN_VISIBILITY_DEFAULT})"
  ${HPX_HIDDEN_VISIBILITY_DEFAULT}
  ADVANCED
)

hpx_option(
  HPX_WITH_LOGGING BOOL "Build HPX with logging enabled (default: ON)." ON
  ADVANCED
)

hpx_option(
  HPX_WITH_DISABLED_SIGNAL_EXCEPTION_HANDLERS
  BOOL
  "Disables the mechanism that produces debug output for caught signals and unhandled exceptions (default: OFF)"
  OFF
  ADVANCED
)
if(HPX_WITH_DISABLED_SIGNAL_EXCEPTION_HANDLERS)
  hpx_add_config_define(HPX_HAVE_DISABLED_SIGNAL_EXCEPTION_HANDLERS)
endif()

hpx_option(
  HPX_WITH_PAPI BOOL "Enable the PAPI based performance counter." OFF
  CATEGORY "Profiling"
)
if(HPX_WITH_PAPI)
  if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    hpx_error(
      "HPX_WITH_PAPI was set to ON, but PAPI can only be used on Linux (this is ${CMAKE_SYSTEM_NAME})"
    )
  endif()
  hpx_add_config_define(HPX_HAVE_PAPI)
endif()

# AGAS related build options
hpx_option(
  HPX_WITH_AGAS_DUMP_REFCNT_ENTRIES BOOL
  "Enable dumps of the AGAS refcnt tables to logs (default: OFF)" OFF
  CATEGORY "AGAS"
  ADVANCED
)
if(HPX_WITH_AGAS_DUMP_REFCNT_ENTRIES)
  hpx_add_config_define(HPX_HAVE_AGAS_DUMP_REFCNT_ENTRIES)
endif()

# Should networking be supported?
hpx_option(
  HPX_WITH_NETWORKING BOOL
  "Enable support for networking and multi-node runs (default: ON)" ON
  CATEGORY "Parcelport"
)

if(HPX_WITH_NETWORKING)
  hpx_add_config_define(HPX_HAVE_NETWORKING)

  # Parcelport related build options
  set(_parcel_profiling_default OFF)
  if(HPX_WITH_APEX)
    set(_parcel_profiling_default ON)
  endif()

  hpx_option(
    HPX_WITH_PARCEL_PROFILING BOOL "Enable profiling data for parcels"
    ${_parcel_profiling_default}
    CATEGORY "Parcelport"
    ADVANCED
  )

  if(HPX_WITH_PARCEL_PROFILING)
    hpx_add_config_define(HPX_HAVE_PARCEL_PROFILING)
  endif()

  # Parcelport related build options and #define in global defines.hpp
  hpx_option(
    HPX_WITH_PARCELPORT_LIBFABRIC
    BOOL
    "Enable the libfabric based parcelport. This is currently an experimental feature"
    OFF
    CATEGORY "Parcelport"
    ADVANCED
  )
  if(HPX_WITH_PARCELPORT_LIBFABRIC)
    hpx_add_config_define(HPX_HAVE_PARCELPORT_LIBFABRIC)
  endif()

  hpx_option(
    HPX_WITH_PARCELPORT_MPI BOOL "Enable the MPI based parcelport." OFF
    CATEGORY "Parcelport"
  )
  if(HPX_WITH_PARCELPORT_MPI)
    hpx_add_config_define(HPX_HAVE_PARCELPORT_MPI)
    if(NOT HPX_WITH_FETCH_HPXLOCAL AND NOT HPXLocal_WITH_ASYNC_MPI)
      hpx_error(
        "HPX is configured with the MPI parcelport (HPX_WITH_PARCELPORT_MPI=ON), but HPXLocal is configured without MPI support (HPXLocal_WITH_ASYNC_MPI=OFF). The MPI parcelport requires HPXLocal_WITH_ASYNC_MPI=ON."
      )
    endif()
  endif()

  hpx_option(
    HPX_WITH_PARCELPORT_TCP BOOL "Enable the TCP based parcelport." ON
    CATEGORY "Parcelport"
  )
  if(HPX_WITH_PARCELPORT_TCP)
    hpx_add_config_define(HPX_HAVE_PARCELPORT_TCP)
  endif()
  hpx_option(
    HPX_WITH_PARCELPORT_ACTION_COUNTERS
    BOOL
    "Enable performance counters reporting parcelport statistics on a per-action basis."
    OFF
    CATEGORY "Parcelport"
  )
  if(HPX_WITH_PARCELPORT_ACTION_COUNTERS)
    hpx_add_config_define(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
  endif()
endif(HPX_WITH_NETWORKING)

# External libraries/frameworks used by some of the examples and benchmarks
hpx_option(
  HPX_WITH_EXAMPLES_HDF5 BOOL
  "Enable examples requiring HDF5 support (default: OFF)." OFF
  CATEGORY "Build Targets"
  ADVANCED
)
if(HPX_WITH_EXAMPLES_HDF5)
  find_package(HDF5 COMPONENTS CXX)
  if(NOT HDF5_FOUND)
    set(HPX_WITH_EXAMPLES_HDF5 OFF)
  endif()
endif()

# Disabling the Qt example on BG/Q as GUIs don't make sense there anyways
if(NOT "${HPX_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
  hpx_option(
    HPX_WITH_EXAMPLES_QT4 BOOL
    "Enable examples requiring Qt4 support (default: OFF)." OFF
    CATEGORY "Build Targets"
    ADVANCED
  )
  if(HPX_WITH_EXAMPLES_QT4)
    find_package(Qt4)
    if(NOT QT4_FOUND)
      set(HPX_WITH_EXAMPLES_QT4 OFF)
    endif()
  endif()
endif()

hpx_option(
  HPX_WITH_TESTS_DEBUG_LOG BOOL
  "Turn on debug logs (--hpx:debug-hpx-log) for tests (default: OFF)" OFF
  CATEGORY "Debugging"
  ADVANCED
)

hpx_option(
  HPX_WITH_TESTS_DEBUG_LOG_DESTINATION STRING
  "Destination for test debug logs (default: cout)" "cout"
  CATEGORY "Debugging"
  ADVANCED
)

hpx_option(
  HPX_WITH_TESTS_MAX_THREADS_PER_LOCALITY
  STRING
  "Maximum number of threads to use for tests (default: 0, use the number of threads specified by the test)"
  0
  CATEGORY "Debugging"
  ADVANCED
)

hpx_option(
  HPX_WITH_PARALLEL_TESTS_BIND_NONE
  BOOL
  "Pass --hpx:bind=none to tests that may run in parallel (cmake -j flag) (default: OFF)"
  OFF
  CATEGORY "Debugging"
  ADVANCED
)

hpx_option(
  HPX_WITH_APEX BOOL "Enable APEX instrumentation support." OFF
  CATEGORY "Profiling"
)

# run hpx_main on all localities by default
hpx_option(
  HPX_WITH_RUN_MAIN_EVERYWHERE BOOL
  "Run hpx_main by default on all localities (default: OFF)." OFF ADVANCED
)
if(HPX_WITH_RUN_MAIN_EVERYWHERE)
  hpx_add_config_define(HPX_HAVE_RUN_MAIN_EVERYWHERE)
endif()

if(HPX_WITH_NETWORKING)
  # Options for our plugins
  hpx_option(
    HPX_WITH_COMPRESSION_BZIP2 BOOL
    "Enable bzip2 compression for parcel data (default: OFF)." OFF ADVANCED
  )
  hpx_option(
    HPX_WITH_COMPRESSION_SNAPPY BOOL
    "Enable snappy compression for parcel data (default: OFF)." OFF ADVANCED
  )
  hpx_option(
    HPX_WITH_COMPRESSION_ZLIB BOOL
    "Enable zlib compression for parcel data (default: OFF)." OFF ADVANCED
  )

  # Parcel coalescing is used by the main HPX library, enable it always
  hpx_option(
    HPX_WITH_PARCEL_COALESCING BOOL
    "Enable the parcel coalescing plugin (default: ON)." ON ADVANCED
  )
  if(HPX_WITH_PARCEL_COALESCING)
    hpx_add_config_define(HPX_HAVE_PARCEL_COALESCING)
    # Adaptive parcel coalescing related metrics counters are enabled only if
    # both parcel coalescing plugin and thread idle rate counters are enabled.
  endif()
endif()

# Developer tools
hpx_option(
  HPX_WITH_VIM_YCM BOOL
  "Generate HPX completion file for VIM YouCompleteMe plugin" OFF ADVANCED
)

# ##############################################################################
# Special diagnostic flags, do not enable by default, only if needed
hpx_option(
  HPX_WITH_CHECK_MODULE_DEPENDENCIES
  BOOL
  "Verify that no modules are cross-referenced from a different module category \
  (default: OFF)"
  OFF
  ADVANCED
)

# Check for compiler compatibility
#

# Check if the selected compiler versions are supposed to work with our codebase
if(CMAKE_COMPILER_IS_GNUCXX AND HPX_WITH_GCC_VERSION_CHECK)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    hpx_error(
      "GCC 7.0 or higher is required. Specify HPX_WITH_GCC_VERSION_CHECK=OFF to ignore this error."
    )
  endif()
endif()

if(MSVC)
  if(MSVC_VERSION LESS 1900)
    hpx_error("MSVC x64 2015 or higher is required.")
  elseif(NOT CMAKE_CL_64)
    hpx_warn(
      "MSVC (32Bit) will compile but will fail running larger applications because of limitations in the Windows OS."
    )
  endif()
endif()

# Setup platform specific compiler options and check for compatible compilers
if("${HPX_PLATFORM_UC}" STREQUAL "NATIVE")
  hpx_info("Compiling with the native toolset")
elseif("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
  hpx_info("Compiling for Android devices")
elseif("${HPX_PLATFORM_UC}" STREQUAL "XEONPHI")
  hpx_info("Compiling for Intel Xeon Phi devices")
  if(NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
    hpx_error("HPX on the MIC can only be compiled with the Intel compiler.")
  endif()
elseif("${HPX_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    hpx_error("HPX on the BG/Q can only be compiled with bgclang")
  endif()
  hpx_info("Compiling for BlueGene/Q")
endif()

# ##############################################################################
# Some special handling of the compilation is need on build infrastructure for
# generating packages for target architecture, see issue #3575

hpx_option(
  HPX_WITH_BUILD_BINARY_PACKAGE
  BOOL
  "Build HPX on the build infrastructure on any LINUX distribution (default: OFF)."
  OFF
  ADVANCED
)

if(NOT WIN32)
  # ############################################################################
  # Macro definitions for system headers
  # ############################################################################
  add_definitions(-D_GNU_SOURCE)

  # ############################################################################
  # System libraries
  # ############################################################################
  if(NOT MSVC)
    if(HPX_CXX11_STD_ATOMIC_LIBRARIES)
      target_link_libraries(
        hpx_base_libraries INTERFACE ${HPX_CXX11_STD_ATOMIC_LIBRARIES}
      )
    endif()
  endif()

  if(HPX_HAVE_LIBSUPCPP)
    target_link_libraries(hpx_base_libraries INTERFACE supc++)
  endif()

  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
    target_link_libraries(hpx_base_libraries INTERFACE dl)
  endif()

  if(NOT APPLE AND NOT ("${HPX_PLATFORM_UC}" STREQUAL "ANDROID"))
    target_link_libraries(hpx_base_libraries INTERFACE rt)
  endif()

  if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
    target_link_libraries(hpx_base_libraries INTERFACE log)
  endif()

  if(APPLE)
    hpx_add_compile_flag_if_available(-ftemplate-depth=256)
  endif()
endif()

if(WIN32)
  if(MSVC)
    hpx_add_target_compile_option(-Ox PUBLIC CONFIGURATIONS Release)

    # even VS2017 has an ICE when compiling with -Ob2
    hpx_add_target_compile_option(-Ob1 PUBLIC CONFIGURATIONS Release)

    # VS2012 and above has a special flag for improving the debug experience by
    # adding more symbol information to the build (-d2Zi)
    hpx_add_target_compile_option(-d2Zi+ PUBLIC CONFIGURATIONS RelWithDebInfo)

    # VS2013 and above know how to do link time constant data segment folding
    # VS2013 update 2 and above know how to remove debug information for
    # non-referenced functions and data (-Zc:inline)
    hpx_add_target_compile_option(-Zc:inline PUBLIC)
    hpx_add_target_compile_option(
      -Gw PUBLIC CONFIGURATIONS Release RelWithDebInfo MinSizeRel
    )
    hpx_add_target_compile_option(-Zo PUBLIC CONFIGURATIONS RelWithDebInfo)

    # Exceptions
    hpx_add_target_compile_option(-EHsc)
    if(MSVC_VERSION GREATER_EQUAL 1900)
      # assume conforming (throwing) operator new implementations
      hpx_add_target_compile_option(-Zc:throwingNew PUBLIC)

      # enable faster linking (requires VS2015 Update1) disabled for now as this
      # flag crashes debugger hpx_remove_link_flag(/debug CONFIGURATIONS Debug)
      # hpx_add_link_flag(/debug:fastlink CONFIGURATIONS Debug)

      # Update 2 requires to set _ENABLE_ATOMIC_ALIGNMENT_FIX for it to compile
      # atomics
      hpx_add_config_cond_define(_ENABLE_ATOMIC_ALIGNMENT_FIX)

      # Update 3 allows to flag rvalue misuses and enforces strict string const-
      # qualification conformance
      hpx_add_target_compile_option(-Zc:rvalueCast PUBLIC)
      hpx_add_target_compile_option(-Zc:strictStrings PUBLIC)

      # Update 8 requires to set _ENABLE_EXTENDED_ALIGNED_STORAGE for it to
      # compile atomics
      hpx_add_config_cond_define(_ENABLE_EXTENDED_ALIGNED_STORAGE)

      # Make sure that __cplusplus is properly defined
      hpx_add_target_compile_option(-Zc:__cplusplus PUBLIC)

      # Silence C++17 deprecation warnings
      hpx_add_config_cond_define(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

      # Silence C++20 deprecation warnings
      hpx_add_config_cond_define(_SILENCE_ALL_CXX20_DEPRECATION_WARNINGS)

      # ASan is available in Visual Studion starting V16.8
      if((MSVC_VERSION GREATER_EQUAL 1928) AND HPXLocal_WITH_SANITIZERS)
        hpx_add_target_compile_option(
          -fsanitize=address PUBLIC CONFIGURATIONS RelWithDebInfo Debug
        )
      endif()
    endif()

    # Runtime type information
    hpx_add_target_compile_option(-GR PUBLIC)
    # Multiprocessor build
    hpx_add_target_compile_option(-MP PUBLIC)
    # Increase the maximum size of object file sections
    hpx_add_target_compile_option(-bigobj PUBLIC)
  endif()

  target_link_libraries(hpx_base_libraries INTERFACE psapi shlwapi)

  # ############################################################################
  # Macro definitions for system headers
  # ############################################################################
  add_definitions(-D_WINDOWS)
  add_definitions(-D_WIN32)
  hpx_add_config_cond_define(_WIN32_WINNT 0x0602)
  hpx_add_config_cond_define(_SCL_SECURE_NO_WARNINGS)
  hpx_add_config_cond_define(_CRT_SECURE_NO_WARNINGS)
  hpx_add_config_cond_define(_SCL_SECURE_NO_DEPRECATE)
  hpx_add_config_cond_define(_CRT_SECURE_NO_DEPRECATE)
  hpx_add_config_cond_define(_CRT_NONSTDC_NO_WARNINGS)
  hpx_add_config_cond_define(_WINSOCK_DEPRECATED_NO_WARNINGS)
  hpx_add_config_cond_define(_CRT_NON_CONFORMING_SWPRINTFS)
  hpx_add_config_cond_define(_SILENCE_FPOS_SEEKPOS_DEPRECATION_WARNING)

  # ############################################################################
  # Boost
  # ############################################################################

  hpx_add_config_cond_define(BOOST_USE_WINDOWS_H)
  if(NOT CMAKE_CL_64)
    hpx_add_config_cond_define(BOOST_NO_ALIGNMENT)
  endif()
  hpx_add_config_cond_define(PSAPI_VERSION 1)
endif()

# Configure Warnings
if(HPX_WITH_COMPILER_WARNINGS)
  if(MSVC) # Adding special warning settings for the MSVC compiler ...
    hpx_add_compile_flag(-W3)

    # MSVC2012/2013 are overeager to report 'qualifier applied to function type
    # has no meaning; ignored'
    hpx_add_compile_flag(-wd4180)

    # Boost.Lockfree triggers 'warning C4307: '+' : integral constant overflow'
    # which is benign
    hpx_add_compile_flag(-wd4307)

    # object allocated on the heap may not be aligned
    hpx_add_compile_flag(-wd4316)

    # max symbol length exceeded
    hpx_add_compile_flag(-wd4503)

    # 'int': forcing value to bool 'true' or 'false' (performance warning)
    hpx_add_compile_flag(-wd4800)

    # vcpkg enables the /utf-8 option which causes (benign) warnings in the
    # Spirit headers: The file contains a character starting at offset ... that
    # is illegal in the current source character set
    hpx_add_compile_flag(-wd4828)
  else() # Trial and error approach for any other compiler ...
    hpx_add_compile_flag_if_available(-Wall)
    hpx_add_compile_flag_if_available(-Wextra)
    hpx_add_compile_flag_if_available(-Wpedantic)
    hpx_add_compile_flag_if_available(-Wno-strict-aliasing)
    hpx_add_compile_flag_if_available(-Wno-sign-promo)
    hpx_add_compile_flag_if_available(-Wno-attributes)
    hpx_add_compile_flag_if_available(-Wno-cast-align)

    # We do not in general guarantee ABI compatibility between C++ standards, so
    # we ignore this warning
    hpx_add_compile_flag_if_available(-Wno-noexcept-type)

    # Be extra strict about format checks Boost.Logging is built on fprintf,
    # sadly
    hpx_add_compile_flag_if_available(-Wformat=2)
    hpx_add_compile_flag_if_available(-Wno-format-nonliteral)

    # Self initialization is dangerous
    hpx_add_compile_flag_if_available(-Winit-self)

    # For portability
    hpx_add_compile_flag_if_available(-Wdouble-promotion)

    # Warn about casting that violates qualifiers or alignment
    hpx_add_compile_flag_if_available(-Wcast-qual)
    if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      # Clang is overeager in reporting cast alignment problems in Boost
      hpx_add_compile_flag_if_available(-Wcast-align)
    endif()

    # False positive when build with Vc and Clang
    if(NOT (HPXLocal_WITH_DATAPAR_VC AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL
                                         "Clang")
    )
      hpx_add_compile_flag_if_available(-Werror=trampolines)
    endif()
    hpx_add_compile_flag_if_available(-Werror=parentheses)
    hpx_add_compile_flag_if_available(-Werror=reorder)
    hpx_add_compile_flag_if_available(-Werror=return-type)
    hpx_add_compile_flag_if_available(-Werror=sequence-point)
    hpx_add_compile_flag_if_available(-Werror=uninitialized)
    hpx_add_compile_flag_if_available(-Werror=format)
    hpx_add_compile_flag_if_available(-Werror=missing-braces)
    hpx_add_compile_flag_if_available(-Werror=sign-compare)
  endif()
endif()

# Configure compiler warnings as errors
if(HPX_WITH_COMPILER_WARNINGS_AS_ERRORS)
  if(MSVC)
    hpx_add_compile_flag(-WX)
  else()
    hpx_add_compile_flag_if_available(-Werror)
  endif()
endif()

# Diagnostics
if(MSVC)
  # Display full paths in diagnostics
  hpx_add_compile_flag(-FC)
else()
  # Show the flags that toggle each warning
  hpx_add_compile_flag_if_available(-fdiagnostics-show-option)

  # VLAs are a GNU extensions that we forbid as they are not supported on MSVC
  hpx_add_compile_flag_if_available(-Werror=vla)
  # No return statement in a non-void function can lead to garbage return values
  # in GCC.
  hpx_add_compile_flag_if_available(-Werror=return-type)

  # We get false positives all over the place with this.
  if(CMAKE_COMPILER_IS_GNUCXX)
    hpx_add_compile_flag_if_available(-Wno-unused-but-set-parameter)
    hpx_add_compile_flag_if_available(-Wno-unused-but-set-variable)
    # Uninitialized variables are bad, earlier compilers issue spurious warnings
    hpx_add_compile_flag_if_available(-Werror=uninitialized)
    hpx_add_compile_flag_if_available(-Wno-unused-local-typedefs)
    # -Werror=maybe-uninitialized leads to false positives.
    hpx_add_compile_flag_if_available(-Wno-maybe-uninitialized)
  endif()

  # False positive when building with Vc and Clang
  if(NOT (HPXLocal_WITH_DATAPAR_VC AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL
                                       "Clang")
  )
    # Silence warning about __sync_fetch_and_nand changing semantics
    hpx_add_compile_flag_if_available(-Wno-sync-nand)
  endif()

  # Silence warnings about deleting polymorphic objects with non-virtual dtors.
  # These come from within Boost.
  if(CMAKE_COMPILER_IS_GNUCXX)
    hpx_add_compile_flag_if_available(-Wno-delete-non-virtual-dtor)
  endif()

  # Check if our libraries have unresolved symbols
  if(NOT APPLE
     AND NOT WIN32
     AND NOT HPXLocal_WITH_SANITIZERS
  )
    hpx_add_link_flag_if_available(
      -Wl,-z,defs # TARGETS SHARED_LIBRARY EXECUTABLE
    )
  endif()
  if(WIN32)
    target_link_libraries(hpx_base_libraries INTERFACE psapi WS2_32 mswsock)
  endif()

  if("${HPX_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
    hpx_add_compile_flag_if_available(-Wno-deprecated-register)
  endif()

  if(HPX_WITH_HIDDEN_VISIBILITY)
    hpx_add_compile_flag_if_available(-fvisibility=hidden)
    hpx_add_link_flag_if_available(-fvisibility=hidden)
    hpx_add_config_define(HPX_HAVE_ELF_HIDDEN_VISIBILITY)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    hpx_add_compile_flag_if_available(-Wno-cast-align)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # Disable the following warnings: #1170: invalid redeclaration of nested
    # class
    hpx_add_compile_flag_if_available(-wd1170)
    # #858: type qualifier on return type is meaningless
    hpx_add_compile_flag_if_available(-wd858)
    # #1098: the qualifier on this friend declaration is ignored
    hpx_add_compile_flag_if_available(-wd1098)
    # #488: template parameter not used in declaring the parameter type
    hpx_add_compile_flag_if_available(-wd488)
    # #2203: cast discards qualifiers from target type (needed for mvapich2 mpi
    # header)
    hpx_add_compile_flag_if_available(-wd2203)
    # #2536: cannot specify explicit initializer for arrays
    hpx_add_compile_flag_if_available(-wd2536)
    # #1286: invalid attribute
    hpx_add_compile_flag_if_available(-wd1286)
  endif()
endif()

# ##############################################################################
# Find Our dependencies: These are all dependencies needed to build the core
# library. Dependencies that are only needed by plugins, examples or tests
# should be found separately in the appropriate subdirectory.
include(HPX_SetupThreads)
include(HPX_SetupPapi)
include(HPX_SetupHPXLocal)

# Setup our required Boost libraries.
include(HPX_SetupBoost)
include(HPX_SetupBoostIostreams)

# reset external source lists
add_hpx_library_sources_noglob(hpx_external)
add_hpx_library_headers_noglob(hpx_external)

# Setup packages and subprojects
include(HPX_SetupApex)

if(HPX_WITH_VIM_YCM)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# ##############################################################################
if(APPLE
   OR MSVC
   OR HPXLocal_WITH_GPU_SUPPORT
)
  set(HPX_WITH_PKGCONFIG
      OFF
      CACHE INTERNAL "" FORCE
  )
endif()

if(HPX_WITH_LOGGING)
  if(NOT HPXLocal_WITH_LOGGING)
    hpx_error("HPX_WITH_LOGGING=ON requires HPXLocal_WITH_LOGGING=ON")
  endif()
  hpx_add_config_define(HPX_HAVE_LOGGING)
endif()

if(HPX_WITH_APEX)
  if(NOT HPXLocal_WITH_APEX)
    hpx_error("HPX_WITH_APEX=ON requires HPXLocal_WITH_APEX=ON")
  endif()
  hpx_add_config_define(HPX_HAVE_APEX)
endif()

# ##############################################################################
# HPX_PREFIX The prefix is the default search path for HPX plugins
# ##############################################################################
if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
  set(HPX_PREFIX "lib")
  set(HPX_BUILD_PREFIX "lib")
else()
  set(HPX_PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(HPX_BUILD_PREFIX "${PROJECT_BINARY_DIR}")
endif()

# Note: on windows systems the ':' will be converted to a ';' at runtime
hpx_add_config_cond_define(
  HPX_DEFAULT_COMPONENT_PATH_SUFFIXES
  "\"/${CMAKE_INSTALL_LIBDIR}/hpx:/${CMAKE_INSTALL_BINDIR}/hpx:/lib/hpx:/bin/hpx\""
)

# ##############################################################################
# search path configuration
# ##############################################################################
if(HPX_WITH_FULL_RPATH)
  include(HPX_SetFullRPATH)
endif()

# ##############################################################################
# Git commit detection
# ##############################################################################
include(HPX_GitCommit)
hpx_add_config_define(HPX_HAVE_GIT_COMMIT "\"${HPX_WITH_GIT_COMMIT}\"")

hpx_include(SetOutputPaths)

# ##############################################################################
# Add custom targets for tests
# ##############################################################################
if(HPX_WITH_TESTS)
  add_hpx_pseudo_target(tests)
  if(HPX_WITH_TESTS_UNIT)
    add_hpx_pseudo_target(tests.unit)
    add_hpx_pseudo_dependencies(tests tests.unit)
  endif()
  if(HPX_WITH_TESTS_REGRESSIONS)
    add_hpx_pseudo_target(tests.regressions)
    add_hpx_pseudo_dependencies(tests tests.regressions)
  endif()
  if(HPX_WITH_TESTS_BENCHMARKS)
    add_hpx_pseudo_target(tests.performance)
    add_hpx_pseudo_dependencies(tests tests.performance)
  endif()
  if(HPX_WITH_TESTS_HEADERS)
    add_hpx_pseudo_target(tests.headers)
    add_hpx_pseudo_dependencies(tests tests.headers)
  endif()
  if(HPX_WITH_EXAMPLES AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_pseudo_target(tests.examples)
    add_hpx_pseudo_dependencies(tests tests.examples)
  endif()

  enable_testing()
  include(CTest)
endif()

if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples)
endif()

# ##############################################################################
# Debug library postfix
# ##############################################################################
set(CMAKE_DEBUG_POSTFIX "d")
set(HPX_DEBUG_POSTFIX "d")

if(HPX_WITH_COMPRESSION_BZIP2)
  hpx_add_config_define(HPX_HAVE_COMPRESSION_BZIP2)
endif()
if(HPX_WITH_COMPRESSION_SNAPPY)
  hpx_add_config_define(HPX_HAVE_COMPRESSION_SNAPPY)
endif()
if(HPX_WITH_COMPRESSION_ZLIB)
  hpx_add_config_define(HPX_HAVE_COMPRESSION_ZLIB)
endif()

# ##############################################################################
# Add core dependency
# ##############################################################################
add_hpx_pseudo_target(core)

# ##############################################################################
# Add libraries
# ##############################################################################
add_subdirectory(src)
add_subdirectory(libs)

# ##############################################################################
# Configure compression and other plugins
# ##############################################################################
set(HPX_STATIC_PARCELPORT_PLUGINS
    ""
    CACHE INTERNAL "" FORCE
)
add_hpx_pseudo_target(plugins)
add_subdirectory(plugins)

# ##############################################################################
# Add static init and wrap library
# ##############################################################################
add_subdirectory(init)
add_subdirectory(wrap)

# ##############################################################################
# Documentation toolchain (Sphinx, Doxygen, Breathe)
# ##############################################################################
hpx_include(Documentation)

# ##############################################################################
# Target specification
# ##############################################################################
if(HPX_WITH_TOOLS OR HPX_WITH_TESTS_BENCHMARKS)
  add_hpx_pseudo_target(tools)
  add_subdirectory(tools)
endif()

# ##############################################################################
# Add core configuration
# ##############################################################################

# ##############################################################################
# Add components
# ##############################################################################
add_subdirectory(components)

# ##############################################################################
# Tests
# ##############################################################################
if(HPX_WITH_TESTS)
  find_package(PythonInterp)

  if(NOT PYTHONINTERP_FOUND)
    hpx_warn(
      "A python interpreter could not be found. The test suite can not be run automatically."
    )
  endif()

  # pseudo_target added above
  add_subdirectory(tests)
endif()

if(HPX_WITH_EXAMPLES)
  add_subdirectory(examples)
endif()

if(HPX_WITH_DOCUMENTATION)
  add_subdirectory(docs)
endif()

# Configure hpxrun.py
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/hpxrun.py.in"
  "${PROJECT_BINARY_DIR}/bin/hpxrun.py" @ONLY
)

# Set up precompiled headers
if(HPX_WITH_PRECOMPILED_HEADERS)
  set(system_precompiled_headers
      <algorithm>
      <array>
      <atomic>
      <bitset>
      <cassert>
      <cctype>
      <cerrno>
      <chrono>
      <climits>
      <cmath>
      <complex>
      <condition_variable>
      <cstddef>
      <cstdint>
      <cstdio>
      <cstdlib>
      <cstring>
      <ctime>
      <deque>
      <exception>
      <execution>
      <forward_list>
      <fstream>
      <functional>
      <iomanip>
      <ios>
      <iosfwd>
      <iostream>
      <iterator>
      <limits>
      <list>
      <locale>
      <map>
      <memory>
      <mutex>
      <new>
      <numeric>
      <ostream>
      <random>
      <regex>
      <set>
      <shared_mutex>
      <sstream>
      <stack>
      <stdexcept>
      <string>
      <system_error>
      <thread>
      <tuple>
      <type_traits>
      <typeinfo>
      <unordered_map>
      <unordered_set>
      <utility>
      <variant>
      <vector>
      <hwloc.h>
      <asio/basic_waitable_timer.hpp>
      <asio/io_context.hpp>
      <asio/ip/tcp.hpp>
      <boost/config.hpp>
      <boost/fusion/include/vector.hpp>
      <boost/intrusive/slist.hpp>
      <boost/lockfree/queue.hpp>
      <boost/optional.hpp>
      <boost/spirit/home/x3.hpp>
      <boost/tokenizer.hpp>
      <boost/utility/string_ref.hpp>
  )

  if(HPX_WITH_CXX17_FILESYSTEM)
    list(APPEND system_precompiled_headers <filesystem>)
  endif()

  set(system_precompiled_headers_dependencies
      hpx_dependencies_boost hpx_private_flags hpx_public_flags
  )

  target_link_libraries(
    hpx_precompiled_headers
    PRIVATE hpx_public_flags hpx_private_flags hpx_base_libraries
            ${system_precompiled_headers_dependencies}
  )
  target_precompile_headers(
    hpx_precompiled_headers PRIVATE ${system_precompiled_headers}
  )

  # Headers that should be precompiled for things depending on HPX (executables,
  # libraries).
  set(hpx_precompiled_headers_modules ${HPX_ENABLED_MODULES})
  # The init_runtime modules cannot be precompiled because they use macros that
  # can depend on the application (HPX_PREFIX and HPX_APPLICATION_NAME_DEFAULT).
  list(REMOVE_ITEM hpx_precompiled_headers_modules init_runtime
       init_runtime_local
  )
  # config, version, and the include modules don't have hpx/module headers.
  list(REMOVE_ITEM hpx_precompiled_headers_modules config include include_local
       version
  )
  list(TRANSFORM hpx_precompiled_headers_modules PREPEND "<hpx/modules/")
  list(TRANSFORM hpx_precompiled_headers_modules APPEND ".hpp>")

  # We do not link to HPX::hpx as it will set HPX_APPLICATION_NAME_DEFAULT and
  # HPX_PREFIX to values that don't match those used in tests. We instead set
  # them manually here.
  target_link_libraries(hpx_exe_precompiled_headers PRIVATE hpx_full)
  target_compile_definitions(
    hpx_exe_precompiled_headers PRIVATE "HPX_PREFIX=\"${HPX_BUILD_PREFIX}\""
                                        "HPX_APPLICATION_EXPORTS"
  )
  target_precompile_headers(
    hpx_exe_precompiled_headers PRIVATE ${hpx_precompiled_headers_modules}
  )
endif()

# ##############################################################################
# installation instructions
# ##############################################################################
install(
  FILES "${PROJECT_BINARY_DIR}/bin/hpxrun.py"
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT core
  PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
)

install(
  # install all hpx header files
  DIRECTORY hpx/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpx
  COMPONENT core
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE
)

# Install all HPX header that have been configured using various cmake options
install(
  DIRECTORY "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpx
  COMPONENT core
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE
)

install(
  # Install all HPX cmake utility files
  DIRECTORY cmake/
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
  COMPONENT core
  PATTERN "templates" EXCLUDE
  PATTERN "packaging" EXCLUDE
  PATTERN ".git" EXCLUDE
)

if("${HPX_PLATFORM_UC}" STREQUAL "XEONPHI")
  # FIXME: push changes upstream
  install(
    DIRECTORY external/asio/boost
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpx/external
    COMPONENT core
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN ".git" EXCLUDE
  )
endif()

install(
  FILES "${PROJECT_SOURCE_DIR}/LICENSE_1_0.txt"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/hpx
  COMPONENT license
)

if(HPX_WITH_DOCUMENTATION)
  install(
    FILES "${PROJECT_SOURCE_DIR}/docs/index.html"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/hpx/
    COMPONENT docs
  )

  if("html" IN_LIST HPX_WITH_DOCUMENTATION_OUTPUT_FORMATS)
    install(
      DIRECTORY "${PROJECT_BINARY_DIR}/share/hpx/docs/html/"
      DESTINATION ${CMAKE_INSTALL_DATADIR}/hpx/docs/html
      COMPONENT docs
      PATTERN "*.buildinfo" EXCLUDE
    )
  endif()

  if("singlehtml" IN_LIST HPX_WITH_DOCUMENTATION_OUTPUT_FORMATS)
    install(
      DIRECTORY "${PROJECT_BINARY_DIR}/share/hpx/docs/singlehtml/"
      DESTINATION ${CMAKE_INSTALL_DATADIR}/hpx/docs/singlehtml
      COMPONENT docs
      PATTERN "*.buildinfo" EXCLUDE
    )
  endif()

  if("latexpdf" IN_LIST HPX_WITH_DOCUMENTATION_OUTPUT_FORMATS)
    install(
      FILES "${PROJECT_BINARY_DIR}/share/hpx/docs/latexpdf/latex/HPX.pdf"
      DESTINATION ${CMAKE_INSTALL_DATADIR}/hpx/docs/pdf/
      COMPONENT docs
    )
  endif()

  if("man" IN_LIST HPX_WITH_DOCUMENTATION_OUTPUT_FORMATS)
    install(
      FILES "${PROJECT_BINARY_DIR}/share/hpx/docs/man/hpx.1"
      DESTINATION ${CMAKE_INSTALL_MANDIR}/
      COMPONENT docs
    )
  endif()
endif()

if(HPX_WITH_VIM_YCM)
  set(build_dir_file ${PROJECT_BINARY_DIR}/.ycm_extra_conf.py)
  set(source_dir_file ${PROJECT_SOURCE_DIR}/.ycm_extra_conf.py)
  configure_file(
    ${PROJECT_SOURCE_DIR}/tools/vim/.ycm_extra_conf.py ${build_dir_file} @ONLY
  )
  add_custom_target(
    configure_ycm
    COMMAND ${CMAKE_COMMAND} -E copy ${build_dir_file} ${source_dir_file}
    COMMENT "Copying YCM config file to source directory"
    VERBATIM
  )
  hpx_info(
    "VIM YouCompleteMe: run 'make configure_ycm' to copy config file to source directory and enable support in YCM. To enable automatic loading of configure file, add to your .vimrc option: \"let g:ycm_extra_conf_globlist = ['${PROJECT_SOURCE_DIR}/*']\""
  )
endif()

# ##############################################################################
# Add rpm packaging

hpx_option(
  HPX_WITH_RPM BOOL "Enable or disable the generation of rpm packages" OFF
  ADVANCED
)

if(HPX_WITH_RPM)
  add_subdirectory(cmake/packaging/rpm)
endif()

# ##############################################################################
# print overall configuration summary
include(HPX_PrintSummary)
create_configuration_summary("Configuration summary:\n--" "hpx")

include(HPX_ExportTargets)
# Modules can't link to this if not exported
install(
  TARGETS hpx_base_libraries
  EXPORT HPXInternalTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT hpx_base_libraries
)
hpx_export_internal_targets(hpx_base_libraries)

# ##############################################################################
# store cache vars and their values in order for them to be forwarded to the
# projects (needs to be before the HPX_GeneratePackage call)
include(HPX_ForwardCacheVariables)

# ##############################################################################
# External build system support (FindHPX.cmake and pkg-config).
include(HPX_GeneratePackage)

message("")
message("HPX will be installed to ${CMAKE_INSTALL_PREFIX}")
message("")
