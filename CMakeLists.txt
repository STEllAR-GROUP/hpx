# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011-2013 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
# Copyright (c)      2013 Jeroen Habraken
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

set(HPX_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)" FORCE)

# Always force CMAKE_CONFIGURATION_TYPES to be the same as CMAKE_BUILD_TYPE
# (at least for now, until we figure out how to use multiple build types in
# the same project).
set(CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPE}" CACHE INTERNAL "Configuration types" FORCE)

################################################################################
# project metadata
################################################################################
project(hpx CXX C)

set(HPX_MAJOR_VERSION 0)
set(HPX_MINOR_VERSION 9)
set(HPX_PATCH_LEVEL   6)
set(HPX_VERSION "${HPX_MAJOR_VERSION}.${HPX_MINOR_VERSION}.${HPX_PATCH_LEVEL}")
set(HPX_SOVERSION ${HPX_MAJOR_VERSION})

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

################################################################################
# CMake configuration
################################################################################
set(HPX_INTERNAL_CMAKE ON)

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

# Decide whether to use internal version of Boost.Chrono
if(${BOOST_MINOR_VERSION} LESS 47)
  hpx_warn("boost" "Using internal version of Boost.Chrono, setting HPX_INTERNAL_CHRONO=ON.")
  set(HPX_INTERNAL_CHRONO ON CACHE BOOL "Use HPX's internal version of Boost.Chrono (default: OFF)" FORCE)
endif()

# Decide whether to use internal version of Boost.Move
if(${BOOST_MINOR_VERSION} LESS 48)
  hpx_warn("boost" "Using internal version of Boost.Move, setting HPX_INTERNAL_MOVE=ON.")
  set(HPX_INTERNAL_MOVE ON CACHE BOOL "Use HPX's internal version of Boost.Move (default: OFF)" FORCE)
endif()

# Decide whether to use Boost.Context for coroutine implementation
if(${BOOST_MINOR_VERSION} GREATER 50)
  if (APPLE)
    hpx_option(HPX_USE_GENERIC_COROUTINE_CONTEXT BOOL "Use Boost.Context as the underlying coroutines context switch implementation (default: ON)." ON ADVANCED)
  else()
    hpx_option(HPX_USE_GENERIC_COROUTINE_CONTEXT BOOL "Use Boost.Context as the underlying coroutines context switch implementation (default: OFF)." OFF ADVANCED)
  endif()
endif()

################################################################################
# search path configuration
################################################################################
hpx_include_directories(${hpx_SOURCE_DIR})
if(HPX_INTERNAL_MOVE)
  hpx_add_definitions(-DHPX_INTERNAL_MOVE)
  hpx_include_directories(${hpx_SOURCE_DIR}/external/move)
endif()
if(${BOOST_MINOR_VERSION} LESS 53)
  hpx_include_directories(${hpx_SOURCE_DIR}/external/atomic)
else()
  set(BOOST_LIBRARIES ${BOOST_LIBRARIES} atomic)
endif()
if(HPX_INTERNAL_CHRONO)
  hpx_add_definitions(-DHPX_INTERNAL_CHRONO)
  hpx_add_definitions(-DBOOST_CHRONO_NO_LIB)
  hpx_include_directories(${hpx_SOURCE_DIR}/external/chrono)
else()
  set(BOOST_LIBRARIES ${BOOST_LIBRARIES} chrono)
endif()
if(HPX_USE_GENERIC_COROUTINE_CONTEXT)
  if(${BOOST_MINOR_VERSION} GREATER 50)
    hpx_add_definitions("-DHPX_HAVE_GENERIC_CONTEXT_COROUTINES=1")
    set(BOOST_LIBRARIES ${BOOST_LIBRARIES} context)
  else()
    hpx_error("boost_version" "Boost V1.51.0 or higher is required to use the Boost.Context library.")
  endif()
endif()
if(${BOOST_MINOR_VERSION} LESS 53)
  hpx_include_directories(${hpx_SOURCE_DIR}/external/lockfree)
endif()
hpx_include_directories(${hpx_SOURCE_DIR}/external/cache)
hpx_include_directories(${hpx_SOURCE_DIR}/external/endian)
if(${BOOST_MINOR_VERSION} LESS 49)
  hpx_include_directories(${hpx_SOURCE_DIR}/external/serialization)
endif()

if(UNIX)
  hpx_link_directories(${CMAKE_BINARY_DIR}/lib/hpx)
else()
  hpx_link_directories(${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib/hpx)
endif()

###############################################################################
# Configure option to automatically set all required settings to use the
# XeonPhi
<<<<<<< HEAD
###############################################################################
hpx_option(HPX_NATIVE_MIC BOOL
    "Build HPX to run natively on the Intel MIC coprocessor (default: OFF)."
    OFF ADVANCED)

if(HPX_NATIVE_MIC)
  if(NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
    hpx_error("HPX on the MIC can only be compiled with the Intel compiler.")
  endif()
  set(HPX_MALLOC "tbbmalloc")
=======
###############################################################################
hpx_option(HPX_NATIVE_MIC BOOL
    "Build HPX to run natively on the Intel MIC coprocessor (default: OFF)."
    OFF ADVANCED)

if(HPX_NATIVE_MIC)
  if(NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
    hpx_error("HPX on the MIC can only be compiled with the Intel compiler.")
  endif()
  set(HPX_MALLOC "tbbmalloc")
endif()

###############################################################################
# Configure upper limit for number of threads, this is important for XeonPhi
###############################################################################
hpx_option(HPX_USE_MORE_THAN_64_THREADS BOOL
    "HPX applications will use more than 64 threads (default: OFF)." OFF ADVANCED)
if(HPX_USE_MORE_THAN_64_THREADS OR HPX_NATIVE_MIC)
  hpx_add_definitions(-DHPX_HAVE_MORE_THAN_64_THREADS)
>>>>>>> master
endif()

###############################################################################
# Configure upper limit for number of threads, this is important for XeonPhi
###############################################################################
hpx_option(HPX_USE_MORE_THAN_64_THREADS BOOL
    "HPX applications will use more than 64 threads (default: OFF)." OFF ADVANCED)
if(HPX_USE_MORE_THAN_64_THREADS OR HPX_NATIVE_MIC)
  hpx_add_definitions(-DHPX_HAVE_MORE_THAN_64_THREADS)
endif()

###############################################################################
# We have a patched version of FindBoost loosely based on the one that Kitware ships
find_package(HPX_Boost)

hpx_include_sys_directories(${BOOST_INCLUDE_DIR})
hpx_link_sys_directories(${BOOST_LIBRARY_DIR})

if(NOT MSVC)
  set(hpx_LIBRARIES ${BOOST_FOUND_LIBRARIES})
endif()

# Boost preprocessor definitions
hpx_add_definitions(-DBOOST_PARAMETER_MAX_ARITY=7)
hpx_add_definitions(-DHPX_COROUTINE_ARG_MAX=1)
if(NOT MSVC)
  hpx_add_definitions(-DHPX_COROUTINE_NO_SEPARATE_CALL_SITES)
endif()
hpx_add_definitions(-DHPX_LOG_NO_TSS)
hpx_add_definitions(-DHPX_LOG_NO_TS)
hpx_add_definitions(-DHPX_PLUGIN_SYMBOLS_PREFIX=hpx)
hpx_add_definitions(-DBOOST_BIGINT_HAS_NATIVE_INT64)

# disable usage of std::atomics in lockfree
if(${BOOST_MINOR_VERSION} LESS 53)
  hpx_add_definitions(-DBOOST_NO_0X_HDR_ATOMIC)
endif()

################################################################################
# Compiler detection code
################################################################################

# C++
hpx_include(GCCVersion)

if(GCC_VERSION AND NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
               AND NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
  hpx_info("gcc_config" "Compiler reports compatibility with GCC version ${GCC_VERSION_STR}")

  hpx_option(HPX_IGNORE_GCC_VERSION BOOL
    "Ignore version reported by gcc (default: OFF)." OFF ADVANCED)

  if(HPX_IGNORE_GCC_VERSION)
    hpx_warn("gcc_config" "GCC 4.4.5 or higher is required. Building HPX will proceed but may fail.")
  elseif(040405 GREATER ${GCC_VERSION})
    hpx_error("gcc_config" "GCC 4.4.5 or higher is required. Specify HPX_IGNORE_GCC_VERSION=ON to overwrite this error.")
  endif()
elseif(MSVC)
  if(NOT (MSVC10 OR MSVC11 OR MSVC12))
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

# If we are compiling with an intel compiler, we need to override the default
# boost compiler config in order to have proper C++11 detection for more
# recent intel compiler versions (>12.0)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    hpx_add_definitions(-DBOOST_COMPILER_CONFIG="${CMAKE_SOURCE_DIR}/hpx/config/boost/compiler/intel.hpp")
endif()


hpx_include(CompilerFlags)

################################################################################
# Git commit detection
################################################################################
include(HPX_GitCommit)

################################################################################
# Installation configuration
################################################################################
# FIXME: Three separate doc strings are used for the *_PREFIX macros.

hpx_option(HPX_DEFAULT_BUILD_TARGETS BOOL
  "Associate the core HPX library with the default build target (default: ON)." ON ADVANCED)

hpx_option(HPX_NO_INSTALL BOOL
  "Build hpx applications so that they can be run without installation (default: OFF)." OFF ADVANCED)

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
# Decide whether to use shared memory based parcelport (starting with Boost V1.52)
################################################################################
if(${BOOST_MINOR_VERSION} GREATER 51)
  hpx_option(HPX_USE_PARCELPORT_SHMEM BOOL "Enable parcelport based on shared memory (default: OFF)." OFF ADVANCED)
  if(HPX_USE_PARCELPORT_SHMEM)
    hpx_add_definitions(-DHPX_HAVE_PARCELPORT_SHMEM)
  endif()
endif()

################################################################################
# Decide whether to use rdma based ibverbs parcelport
################################################################################
hpx_option(HPX_USE_PARCELPORT_IBVERBS BOOL "Enable parcelport based on rdma ibverbs operations (default: OFF)." OFF ADVANCED)
if(HPX_USE_PARCELPORT_IBVERBS)
  find_package(HPX_Ibverbs)
  find_package(HPX_Rdmacm)
  if(IBVERBS_FOUND AND RDMACM_FOUND)
    hpx_add_definitions(-DHPX_HAVE_PARCELPORT_IBVERBS)
  endif()
endif()

################################################################################
# Logging configuration
################################################################################
hpx_option(HPX_NO_LOGGING BOOL "Build hpx with logging completely disabled (default: OFF)." OFF ADVANCED)
if(HPX_NO_LOGGING)
  hpx_add_definitions("-DHPX_NO_LOGGING=1")
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

#add_definitions("-DHPX_PREFIX=\"${HPX_PREFIX}\"")

# Make sure the specified build type is valid.
if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR
        "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo" OR
        "${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel" OR
        "${CMAKE_BUILD_TYPE}" STREQUAL "Release"))
  hpx_error("build_type" "\"${CMAKE_BUILD_TYPE}\" is not a valid build type, must be Debug, Release, RelWithDebInfo, or MinSizeRel.")
endif()

hpx_include(SetOutputPaths)

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
hpx_option(HPX_HAVE_CXX11 BOOL "Use C++11 support, if available (default: ON)" ON ADVANCED)

if(HPX_HAVE_CXX11)
  hpx_add_definitions(-DHPX_HAVE_CXX11)
endif()

################################################################################
# Additional debug support
################################################################################
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  hpx_option(HPX_HAVE_VERIFY_LOCKS BOOL "Enable lock verification code (default: ON)" ON ADVANCED)
else()
  hpx_option(HPX_HAVE_VERIFY_LOCKS BOOL "Enable lock verification code (default: OFF)" OFF ADVANCED)
endif()

if(HPX_HAVE_VERIFY_LOCKS)
  hpx_add_definitions(-DHPX_HAVE_VERIFY_LOCKS=1)
else()
  hpx_add_definitions(-DHPX_HAVE_VERIFY_LOCKS=0)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  hpx_option(HPX_THREAD_DEBUG_INFO BOOL "Enable thread debugging information (default: ON)" ON ADVANCED)
else()
  hpx_option(HPX_THREAD_DEBUG_INFO BOOL "Enable thread debugging information (default: OFF)" OFF ADVANCED)
endif()

if(NOT MSVC)
  hpx_option(HPX_THREAD_GUARD_PAGE BOOL "Enable thread guard page (default: ON)" ON ADVANCED)
  if(HPX_THREAD_GUARD_PAGE)
    hpx_add_definitions(-DHPX_THREAD_GUARD_PAGE=1)
  else()
    hpx_add_definitions(-DHPX_THREAD_GUARD_PAGE=0)
  endif()
endif()
  
if(NOT MSVC) 
  hpx_option(HPX_USE_MMAP BOOL "Use mmap for stack allocation (default: ON)" ON ADVANCED)
  if(HPX_USE_MMAP)
    hpx_add_definitions(-DHPX_USE_MMAP)
  endif()
endif()

if(HPX_THREAD_DEBUG_INFO)
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_PARENT_REFERENCE=1)
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_PHASE_INFORMATION=1)
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_DESCRIPTION=1)
else()
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_PARENT_REFERENCE=0)
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_PHASE_INFORMATION=0)
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_DESCRIPTION=0)
endif()

hpx_option(HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION BOOL "Enable thread stack back trace being captured on suspension (default: OFF)" OFF ADVANCED)

if(HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION)
  hpx_option(HPX_THREAD_BACKTRACE_ON_SUSPENSION_DEPTH STRING "Thread stack back trace depth being captured on suspension (default: 5)" "5" ADVANCED)
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION=1)
  hpx_add_definitions(-DHPX_THREAD_BACKTRACE_ON_SUSPENSION_DEPTH=${HPX_THREAD_BACKTRACE_ON_SUSPENSION_DEPTH})
else()
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION=0)
endif()

hpx_option(HPX_THREAD_MAINTAIN_TARGET_ADDRESS BOOL "Enable storing target address in thread for NUMA awareness (default: OFF)" OFF ADVANCED)
if(HPX_THREAD_MAINTAIN_TARGET_ADDRESS)
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_TARGET_ADDRESS=1)
else()
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_TARGET_ADDRESS=0)
endif()

hpx_option(HPX_THREAD_MAINTAIN_QUEUE_WAITTIME BOOL "Enable collecting queue wait times for threads (default: ON)" ON ADVANCED)
if(HPX_THREAD_MAINTAIN_QUEUE_WAITTIME)
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_QUEUE_WAITTIME=1)
else()
  hpx_add_definitions(-DHPX_THREAD_MAINTAIN_QUEUE_WAITTIME=0)
endif()

################################################################################
# PAPI event counters
################################################################################
find_package(HPX_PAPI)

################################################################################
# Apex event counters
################################################################################
find_package(HPX_TAU)
find_package(HPX_APEX)

################################################################################
# hwloc
################################################################################
if(NOT HPX_DISABLE_HWLOC)
  find_package(HPX_Hwloc)
  if(HWLOC_FOUND)
    hpx_include_sys_directories(${HWLOC_INCLUDE_DIR})
    hpx_link_sys_directories(${HWLOC_LIBRARY_DIR})
    hpx_add_definitions(-DHPX_HAVE_HWLOC)
    set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} ${HWLOC_LIBRARY})
  endif()
<<<<<<< HEAD
endif()

################################################################################
# Security, libsodium: P(ortable|ackageable) NaCl
################################################################################
find_package(HPX_Sodium)
if(SODIUM_FOUND)
  hpx_add_definitions(-DHPX_HAVE_SODIUM)

  hpx_include_sys_directories(${SODIUM_INCLUDE_DIR})
  hpx_link_sys_directories(${SODIUM_LIBRARY_DIR})
  if(MSVC)
    hpx_add_definitions(-DSODIUM_STATIC)
  endif()
  set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} ${SODIUM_LIBRARY})

  hpx_option(HPX_HAVE_SECURITY BOOL "Enable enable security options (default: OFF)" OFF ADVANCED)
  if(HPX_HAVE_SECURITY)
    hpx_add_definitions(-DHPX_HAVE_SECURITY)
  endif()
elseif(HPX_HAVE_SECURITY)
  hpx_error("security" "The libsodium library has to be available to enable security features.")
=======
>>>>>>> master
endif()

###############################################################################
# Configure compression and other plugins
###############################################################################
add_hpx_pseudo_target(plugins)
add_subdirectory(plugins)

if(HPX_USE_COMPRESSION_BZIP2 AND BZIP2_FOUND)
  hpx_add_definitions(-DHPX_HAVE_COMPRESSION_BZIP2)
endif()
if(HPX_USE_COMPRESSION_SNAPPY AND SNAPPY_FOUND)
  hpx_add_definitions(-DHPX_HAVE_COMPRESSION_SNAPPY)
endif()
if(HPX_USE_COMPRESSION_ZLIB AND ZLIB_FOUND)
  hpx_add_definitions(-DHPX_HAVE_COMPRESSION_ZLIB)
endif()

# Parcel coalescing is used by the main HPX library, enable it always
hpx_add_definitions(-DHPX_HAVE_PARCEL_COALESCING)

################################################################################
# Documentation toolchain (DocBook, BoostBook, QuickBook, xsltproc)
################################################################################
hpx_include(Documentation)

################################################################################
# Creating the partially preprocessed header requires a recent version of Wave
################################################################################
hpx_include(Preprocessing)

################################################################################
# Warning configuration
################################################################################
hpx_option(HPX_WARNINGS BOOL "Enable compiler warnings (default: ON)" ON ADVANCED)

################################################################################
# Backtrace configuration
################################################################################
hpx_option(HPX_HAVE_STACKTRACES BOOL "Attach backtraces to HPX exceptions (default: ON)" ON ADVANCED)

if(HPX_HAVE_STACKTRACES OR HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION)
  hpx_info("stacktraces" "Stack traces are enabled.")
  hpx_add_definitions(-DHPX_HAVE_STACKTRACES)
endif()

################################################################################
# Emulation of SwapContext on Windows
################################################################################
if(MSVC)
  hpx_option(HPX_HAVE_SWAP_CONTEXT_EMULATION BOOL "Emulate SwapContext API for coroutines (default: OFF)" OFF ADVANCED)

  if(HPX_HAVE_SWAP_CONTEXT_EMULATION)
    enable_language(ASM_MASM)
    if(CMAKE_ASM_MASM_COMPILER)
      hpx_info("swap_context" "SwitchToFiber emulation is enabled, using compiler: '${CMAKE_ASM_MASM_COMPILER}'")
    else()
      hpx_warn("swap_context" "SwitchToFiber emulation will be disabled, try setting the ASM_MASM environment variable to the assembler executable (ml.exe/ml64.exe)")
      set(HPX_HAVE_SWAP_CONTEXT_EMULATION OFF CACHE BOOL "Emulate SwitchToFiber API for coroutines (default: OFF)" FORCE)
    endif()
  endif()
endif()

################################################################################
# Enable integration with Intel Amplifier and Inspector tools
################################################################################
hpx_option(HPX_HAVE_ITTNOTIFY BOOL "Bind to Intel Amplifier Tool (default: OFF)" OFF ADVANCED)

if(HPX_HAVE_ITTNOTIFY)
  find_package(HPX_Amplifier)
  if(AMPLIFIER_FOUND)
    hpx_info("itt" "Binding to Intel Amplifier tool is enabled.")
    hpx_add_definitions(-DHPX_HAVE_ITTNOTIFY=1)
    hpx_include_sys_directories(${AMPLIFIER_INCLUDE_DIR})
    hpx_link_sys_directories(${AMPLIFIER_LIBRARY_DIR})
    set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} ${AMPLIFIER_LIBRARY})
  else()
    set(HPX_HAVE_ITTNOTIFY OFF FORCE)
    hpx_warn("itt" "Binding to Intel Amplifier tool will be disabled, try setting the AMPLIFIER_ROOT variable to the Intel Amplifier installation directory.")
  endif()
endif()

################################################################################
# Native TLS configuration
################################################################################
if(APPLE)
  hpx_option(HPX_HAVE_NATIVE_TLS BOOL "Use native TLS support if available (default: OFF)" OFF ADVANCED)
else()
  hpx_option(HPX_HAVE_NATIVE_TLS BOOL "Use native TLS support if available (default: ON)" ON ADVANCED)
endif()

if(HPX_HAVE_NATIVE_TLS)
  hpx_info("tls" "Native TLS is enabled.")
  hpx_add_definitions(-DHPX_HAVE_NATIVE_TLS)
endif()

################################################################################
# Disable the use of partially preprocessed header files
################################################################################
hpx_option(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION BOOL "Do not use preprocessed headers (default: OFF)" OFF ADVANCED)

if(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
  hpx_info("ppheaders" "Use of preprocessed headers is disabled.")
  hpx_add_definitions(-DHPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
endif()

################################################################################
# Compiler configuration
################################################################################
# Clear CMake defaults
foreach(language CXX C Fortran)
  set(CMAKE_${language}_FLAGS_DEBUG "" CACHE STRING "Debug flags (${language})" FORCE)
  set(CMAKE_${language}_FLAGS_MINSIZEREL "" CACHE STRING "MinSizeRel flags (${language})" FORCE)
  set(CMAKE_${language}_FLAGS_RELEASE "" CACHE STRING "Release flags (${language})" FORCE)
  set(CMAKE_${language}_FLAGS_RELWITHDEBINFO ""  CACHE STRING "RelWithDebInfo flags (${language})" FORCE)
endforeach()

if(HPX_NATIVE_MIC)
  foreach(language CXX C Fortran)
    set(CMAKE_${language}_FLAGS_DEBUG
        "${CMAKE_${language}_FLAGS_DEBUG} -mmic")
    set(CMAKE_${language}_FLAGS_RELWITHDEBINFO
        "${CMAKE_${language}_FLAGS_RELWITHDEBINFO} -mmic")
    set(CMAKE_${language}_FLAGS_MINSIZEREL
        "${CMAKE_${language}_FLAGS_MINSIZEREL} -mmic")
    set(CMAKE_${language}_FLAGS_RELEASE
        "${CMAKE_${language}_FLAGS_RELEASE} -mmic")
  endforeach()
endif()

if(MSVC)
  add_definitions(-DHPX_BUILD_TYPE="$(Configuration)")
  foreach(language CXX C Fortran)
    set(CMAKE_${language}_FLAGS_DEBUG
        "${CMAKE_${language}_FLAGS_DEBUG} -D_DEBUG")
    set(CMAKE_${language}_FLAGS_RELWITHDEBINFO
        "${CMAKE_${language}_FLAGS_RELWITHDEBINFO} -DNDEBUG -DBOOST_DISABLE_ASSERTS")
    set(CMAKE_${language}_FLAGS_MINSIZEREL
        "${CMAKE_${language}_FLAGS_MINSIZEREL} -DNDEBUG -DBOOST_DISABLE ASSERTS")
    set(CMAKE_${language}_FLAGS_RELEASE
        "${CMAKE_${language}_FLAGS_RELEASE} -DNDEBUG -DBOOST_DISABLE_ASSERTS")
  endforeach()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG -DBOOST_DISABLE_ASSERTS")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -DNDEBUG -DBOOST_DISABLE_ASSERTS")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -DBOOST_DISABLE_ASSERTS")
else()
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    hpx_add_definitions(-DDEBUG)
    hpx_add_definitions(-DHPX_BUILD_TYPE=debug)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    hpx_add_definitions(-DNDEBUG)
    hpx_add_definitions(-DBOOST_DISABLE_ASSERTS)
    hpx_add_definitions(-DHPX_BUILD_TYPE=relwithdebinfo)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    hpx_add_definitions(-DNDEBUG)
    hpx_add_definitions(-DBOOST_DISABLE_ASSERTS)
    hpx_add_definitions(-DHPX_BUILD_TYPE=minsizerel)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    hpx_add_definitions(-DNDEBUG)
    hpx_add_definitions(-DBOOST_DISABLE_ASSERTS)
    hpx_add_definitions(-DHPX_BUILD_TYPE=release)
  endif()
endif()

# Debug library postfix
set(CMAKE_DEBUG_POSTFIX "d")
set(HPX_DEBUG_POSTFIX "d")

################################################################################
# Utility configuration
################################################################################
hpx_option(HPX_UTIL_BIND BOOL "Use hpx::util::bind instead of boost::bind or std::bind (default: ON)" ON ADVANCED)

if(HPX_UTIL_BIND)
  hpx_add_definitions(-DHPX_UTIL_BIND)
endif()

hpx_option(HPX_UTIL_FUNCTION BOOL "Use hpx::util::function instead of boost::function or std::function (default: ON)" ON ADVANCED)

if(HPX_UTIL_FUNCTION)
  hpx_add_definitions(-DHPX_UTIL_FUNCTION)
endif()

hpx_option(HPX_UTIL_TUPLE BOOL "Use hpx::util::tuple instead of boost::tuple or std::tuple (default: ON)" ON ADVANCED)

if(HPX_UTIL_TUPLE)
  hpx_add_definitions(-DHPX_UTIL_TUPLE)
endif()

################################################################################
# Scheduler configuration
################################################################################
hpx_option(HPX_GLOBAL_SCHEDULER BOOL "Enable the use of --queueing=global (default: OFF)" OFF ADVANCED)

if(HPX_GLOBAL_SCHEDULER)
  hpx_add_definitions(-DHPX_GLOBAL_SCHEDULER)
endif()

hpx_option(HPX_LOCAL_SCHEDULER BOOL "Enable the use of --queueing=local (default: OFF)" OFF ADVANCED)

if(HPX_LOCAL_SCHEDULER)
  hpx_add_definitions(-DHPX_LOCAL_SCHEDULER)
endif()

hpx_option(HPX_STATIC_PRIORITY_SCHEDULER BOOL "Enable the use of --queueing=static (default: On)" On ADVANCED)

if(HPX_STATIC_PRIORITY_SCHEDULER)
  hpx_add_definitions(-DHPX_STATIC_PRIORITY_SCHEDULER)
endif()

hpx_option(HPX_ABP_SCHEDULER BOOL "Enable the use of --queueing=abp (default: OFF)" OFF ADVANCED)

if(HPX_ABP_SCHEDULER)
  hpx_add_definitions(-DHPX_ABP_SCHEDULER)
endif()

hpx_option(HPX_ABP_PRIORITY_SCHEDULER BOOL "Enable the use of --queueing=priority_abp (default: OFF)" OFF ADVANCED)

if(HPX_ABP_PRIORITY_SCHEDULER)
  hpx_add_definitions(-DHPX_ABP_PRIORITY_SCHEDULER)
endif()

hpx_option(HPX_HIERARCHY_SCHEDULER BOOL "Enable the use of --queueing=hierarchy (default: OFF)" OFF ADVANCED)

if(HPX_HIERARCHY_SCHEDULER)
  hpx_add_definitions(-DHPX_HIERARCHY_SCHEDULER)
endif()

hpx_option(HPX_PERIODIC_PRIORITY_SCHEDULER BOOL "Enable the use of --queueing=periodic (default: OFF)" OFF ADVANCED)

if(HPX_PERIODIC_PRIORITY_SCHEDULER)
  hpx_add_definitions(-DHPX_PERIODIC_PRIORITY_SCHEDULER)
endif()
  
hpx_option(HPX_GOOGLE_PERFTOOLS BOOL
           "Compile HPX with google perftools (Default: ON)" ON ADVANCED)

if(HPX_GOOGLE_PERFTOOLS)
  find_package(HPX_GooglePerftools)
  if(GOOGLE_PERFTOOLS_FOUND)
    set(hpx_LIBRARIES ${hpx_LIBRARIES} ${GOOGLE_PERFTOOLS_LIBRARY})
    hpx_include_sys_directories(${GOOGLE_PERFTOOLS_INCLUDE_DIR})
    hpx_link_sys_directories(${GOOGLE_PERFTOOLS_LIBRARY_DIR})
    hpx_add_definitions(-DHPX_GOOGLE_PERFTOOLS)
  endif()
endif()

if(NOT HPX_MALLOC)
  set(HPX_MALLOC "TCMalloc")
endif()
set(hpx_MALLOC_LIBRARY "")
hpx_include(SetAllocator)

################################################################################
# MSVC configuration
################################################################################
if(MSVC)
  foreach(language CXX C Fortran)
    set(CMAKE_${language}_FLAGS_DEBUG
        "${CMAKE_${language}_FLAGS_DEBUG} -D_DEBUG -Od -Zi -MDd")
    set(CMAKE_${language}_FLAGS_RELWITHDEBINFO
        "${CMAKE_${language}_FLAGS_RELWITHDEBINFO} -O2 -Zi -MD")
    set(CMAKE_${language}_FLAGS_MINSIZEREL
        "${CMAKE_${language}_FLAGS_MINSIZEREL} -O1 -MD")
    set(CMAKE_${language}_FLAGS_RELEASE
        "${CMAKE_${language}_FLAGS_RELEASE} -Ox -MD -GL")
  endforeach()

  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -RTC1")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -RTC1")

  # Exceptions
  hpx_append_flag(-EHsc)

  # Runtime type information
  hpx_append_flag(-GR)

  # Multiprocessor build
  hpx_append_flag(-MP LANGUAGES CXX C Fortran)

  # Increase the maximum size of object file sections
  hpx_append_flag(-bigobj LANGUAGES CXX C Fortran)

  set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} psapi shlwapi)

  ##############################################################################
  # Stacktraces
  ##############################################################################
  if(HPX_HAVE_STACKTRACES)
    set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} dbghelp)
  endif()

  ##############################################################################
  # C++11
  ##############################################################################
  if(HPX_HAVE_CXX11)
    # C++11 support
    hpx_option(HPX_HAVE_CXX11_RVALUE_REFERENCES BOOL "Compiler supports C++11 rvalue references (default: ON)." ON ADVANCED)
    if(HPX_HAVE_CXX11_RVALUE_REFERENCES)
      hpx_add_definitions(-DHPX_HAVE_CXX11_RVALUE_REFERENCES)
    endif()

    hpx_option(HPX_HAVE_CXX11_LAMBDAS BOOL "Compiler supports C++11 lambda functions (default: ON)." ON ADVANCED)
    if(HPX_HAVE_CXX11_LAMBDAS)
      hpx_add_definitions(-DHPX_HAVE_CXX11_LAMBDAS)
    endif()

    hpx_option(HPX_HAVE_CXX11_AUTO BOOL "Compiler supports C++11 auto keyword (default: ON)." ON ADVANCED)
    if(HPX_HAVE_CXX11_AUTO)
      hpx_add_definitions(-DHPX_HAVE_CXX11_AUTO)
    endif()

    hpx_option(HPX_HAVE_CXX11_DECLTYPE BOOL "Compiler supports C++11 decltype keyword (default: ON)." ON ADVANCED)
    if(HPX_HAVE_CXX11_DECLTYPE)
      hpx_add_definitions(-DHPX_HAVE_CXX11_DECLTYPE)
      hpx_add_definitions(-DBOOST_RESULT_OF_USE_DECLTYPE)
    endif()

    if(MSVC12)
      hpx_option(HPX_HAVE_CXX11_VARIADIC_TEMPLATES BOOL "Compiler supports C++11 variadic templates (default: ON)." ON ADVANCED)
    else()
      hpx_option(HPX_HAVE_CXX11_VARIADIC_TEMPLATES BOOL "Compiler supports C++11 variadic templates (default: OFF)." OFF ADVANCED)
    endif()

    if(HPX_HAVE_CXX11_VARIADIC_TEMPLATES)
      hpx_add_definitions(-DHPX_HAVE_CXX11_VARIADIC_TEMPLATES)
    endif()

    hpx_option(HPX_HAVE_CXX11_UNIQUE_PTR BOOL "Compiler supports C++11 std::unique_ptr (default: ON)." ON ADVANCED)
    if(HPX_HAVE_CXX11_UNIQUE_PTR)
      hpx_add_definitions(-DHPX_HAVE_CXX11_STD_UNIQUE_PTR)
      hpx_add_definitions(-DBOOST_LOCKFREE_HAVE_CXX11_UNIQUE_PTR)
    endif()

    hpx_option(HPX_HAVE_CXX11_STD_TUPLE BOOL "Compiler supports C++11 (movable) std::tuple (default: ON)." ON ADVANCED)
    if(NOT HPX_UTIL_TUPLE AND HPX_HAVE_CXX11_STD_TUPLE)
      hpx_add_definitions(-DHPX_HAVE_CXX11_STD_TUPLE)
    else()
      set(HPX_HAVE_CXX11_STD_TUPLE OFF CACHE BOOL "Compiler supports C++11 (movable) std::tuple." FORCE)
      hpx_info("C++11" "HPX_UTIL_TUPLE is enabled: even while std::tuple is available it will not be used.")
    endif()

    hpx_option(HPX_HAVE_CXX11_STD_BIND  BOOL "Compiler supports C++11 (movable) std::bind (default: OFF)." OFF ADVANCED)
    if(NOT HPX_UTIL_BIND)
      hpx_add_definitions(-DHPX_HAVE_CXX11_STD_BIND)
    else()
      set(HPX_HAVE_CXX11_STD_BIND OFF CACHE BOOL "Compiler supports C++11 (movable) std::bind." FORCE)
      hpx_info("C++11" "HPX_UTIL_BIND is enabled: even while std::bind is available it will not be used.")
    endif()

    hpx_option(HPX_HAVE_CXX11_STD_FUNCTION BOOL "Compiler supports C++11 (movable) std::function (default: OFF)." OFF ADVANCED)
    if(NOT HPX_UTIL_FUNCTION AND HPX_HAVE_CXX11_STD_FUNCTION)
      hpx_add_definitions(-DHPX_HAVE_CXX11_STD_FUNCTION)
    else()
      set(HPX_HAVE_CXX11_STD_FUNCTION OFF CACHE BOOL "Compiler supports C++11 (movable) std::function." FORCE)
      hpx_info("C++11" "HPX_UTIL_FUNCTION is enabled: even while std::function is available it will not be used.")
    endif()
  endif()

  ##############################################################################
  # Diagnostics
  ##############################################################################
  if(HPX_WARNINGS)
    hpx_append_flag(-W3 LANGUAGES C CXX)

    # Boost.Lockfree triggers 'warning C4307: '+' : integral constant overflow'
    # which is benign
    hpx_append_flag(-wd4307)

    if(MSVC11)
      # MSVC2012 is overeager to report 'qualifier applied to function type has no meaning; ignored'
      hpx_append_flag(-wd4180)
    endif()

    # According to the ifort Windows manual, W3 isn't supported
    hpx_append_flag(-W1 LANGUAGES Fortran)
  endif()

  # Display full paths in diagnostics
  hpx_append_flag(-FC LANGUAGES C CXX)

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

  hpx_add_definitions(-DBOOST_USE_WINDOWS_H)
  hpx_add_definitions(-DBOOST_SERIALIZATION_DYN_LINK)
  hpx_add_definitions(-DHPX_HAVE_FIBER_BASED_COROUTINES)
  hpx_add_definitions(-DPSAPI_VERSION=1)

################################################################################
# GCC-compatible compiler configuration
################################################################################
else()
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    # No optimizations
    hpx_use_flag_if_available(-O0 LANGUAGES CXX C Fortran)

    # Generate debugging information
    hpx_use_flag_if_available(-g LANGUAGES CXX C Fortran)

    # Keep the frame pointer
    hpx_use_flag_if_available(-fno-omit-frame-pointer LANGUAGES CXX C Fortran)

    # Don't inline functions
    hpx_use_flag_if_available(-fno-inline LANGUAGES CXX C Fortran)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    # Optimize for speed
    hpx_use_flag_if_available(-O3 LANGUAGES CXX C Fortran)

    # Generate debugging information
    hpx_use_flag_if_available(-g LANGUAGES CXX C Fortran)

    # Don't inline functions
    hpx_use_flag_if_available(-fno-inline LANGUAGES CXX C Fortran)

    # No tail call optimizations
    hpx_use_flag_if_available(-fno-optimize-sibling-calls LANGUAGES CXX C Fortran)

    # Don't allocate registers for wide types (e.g. 128bit ints) independently
    hpx_use_flag_if_available(-fno-split-wide-types LANGUAGES CXX C Fortran)

    # Disable register allocation webs, which make debugging impossible
    hpx_use_flag_if_available(-fno-web LANGUAGES CXX C Fortran)

    # Disable register renaming
    hpx_use_flag_if_available(-fno-rename-registers LANGUAGES CXX C Fortran)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    # Optimize for size
    hpx_use_flag_if_available(-Os LANGUAGES CXX C Fortran)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    # Optimize for speed
    hpx_use_flag_if_available(-O3 LANGUAGES CXX C Fortran)
  endif()

  hpx_option(HPX_PIE BOOL "Compile executables as shared objects" ON ADVANCED)

  # This is necessary to ensure that our executables are ELF DYN objects.
  if(HPX_PIE)
    hpx_use_flag_if_available(-rdynamic LANGUAGES CXX C Fortran)
    hpx_use_flag_if_available(-fPIC LANGUAGES CXX C Fortran)
  endif()

  if(APPLE)
    hpx_use_flag_if_available(-stdlib=libc++ LANGUAGES CXX)
    hpx_use_flag_if_available(-ftemplate-depth=256 LANGUAGES CXX)
  endif()

  ##############################################################################
  # C++11
  ##############################################################################
  if(HPX_HAVE_CXX11)
    # Turn on C++11 support
    hpx_use_flag_if_available(-std=c++0x)

    hpx_check_for_cxx11_rvalue_references(HPX_HAVE_CXX11_RVALUE_REFERENCES
      DEFINITIONS HPX_HAVE_CXX11_RVALUE_REFERENCES)

    hpx_check_for_cxx11_variadic_templates(HPX_HAVE_CXX11_VARIADIC_TEMPLATES
      DEFINITIONS HPX_HAVE_CXX11_VARIADIC_TEMPLATES)

    hpx_check_for_cxx11_lambdas(HPX_HAVE_CXX11_LAMBDAS
      DEFINITIONS HPX_HAVE_CXX11_LAMBDAS)

    hpx_check_for_cxx11_auto(HPX_HAVE_CXX11_AUTO DEFINITIONS HPX_HAVE_CXX11_AUTO)

    hpx_check_for_cxx11_decltype(HPX_HAVE_CXX11_DECLTYPE
      DEFINITIONS HPX_HAVE_CXX11_DECLTYPE BOOST_RESULT_OF_USE_DECLTYPE)

    hpx_check_for_cxx11_std_unique_ptr(HPX_HAVE_CXX11_STD_UNIQUE_PTR
      DEFINITIONS HPX_HAVE_CXX11_STD_UNIQUE_PTR
                  BOOST_LOCKFREE_HAVE_CXX11_STD_UNIQUE_PTR)

    if(NOT HPX_UTIL_TUPLE)
      hpx_check_for_cxx11_std_tuple(HPX_HAVE_CXX11_STD_TUPLE
        DEFINITIONS HPX_HAVE_CXX11_STD_TUPLE)
    else()
      hpx_info("C++11" "HPX_UTIL_TUPLE is enabled: even if std::tuple is available it will not be used.")
    endif()

    if(NOT HPX_UTIL_BIND)
      hpx_check_for_cxx11_std_bind(HPX_HAVE_CXX11_STD_BIND
        DEFINITIONS HPX_HAVE_CXX11_STD_BIND)
    else()
      hpx_info("C++11" "HPX_UTIL_BIND is enabled: even if std::bind is available it will not be used.")
    endif()

    if(NOT HPX_UTIL_FUNCTION)
      hpx_check_for_cxx11_std_function(HPX_HAVE_CXX11_STD_FUNCTION
        DEFINITIONS HPX_HAVE_CXX11_STD_FUNCTION)
    else()
      hpx_info("C++11" "HPX_UTIL_FUNCTION is enabled: even if std::function is available it will not be used.")
    endif()
  endif()

  ##############################################################################
  # Diagnostics
  ##############################################################################
  if(HPX_WARNINGS)
    hpx_use_flag_if_available(-Wall LANGUAGES CXX C Fortran)
    hpx_use_flag_if_available(-Wno-strict-aliasing LANGUAGES CXX C Fortran)
    hpx_use_flag_if_available(-Wsign-promo LANGUAGES CXX)
  endif()



  # VLAs are a GNU extensions that we forbid as they are not supported on MSVC
  hpx_use_flag_if_available(-Werror=vla LANGUAGES CXX C)

  # No return statement in a non-void function can lead to garbage return values
  # in GCC.
  hpx_use_flag_if_available(-Werror=return-type LANGUAGES CXX C)

  # Show the flags that toggle each warning
  hpx_use_flag_if_available(-fdiagnostics-show-option LANGUAGES CXX C Fortran)

  # We get false positives all over the place with this. Also, detection for
  # this flag fails with GCC 4.4 and 4.5.
  if(040600 LESS ${GCC_VERSION} OR 040600 EQUAL ${GCC_VERSION})
    hpx_use_flag_if_available(-Wno-unused-but-set-parameter LANGUAGES CXX C)
    hpx_use_flag_if_available(-Wno-unused-but-set-variable LANGUAGES CXX C)

    # Uninitialized variables are bad, earlier compilers issue spurious warnings
    hpx_use_flag_if_available(-Werror=uninitialized LANGUAGES CXX C)
  endif()

  # Silence warning about __sync_fetch_and_nand changing semantics
  hpx_use_flag_if_available(-Wno-sync-nand LANGUAGES CXX C)

  # Silence warnings about deleting polymorphic objects with non-virtual dtors.
  # These come from within Boost. Detection for this flag fails with GCC 4.4,
  # 4.5 and 4.6.
  if(040700 LESS ${GCC_VERSION} OR 040700 EQUAL ${GCC_VERSION})
    hpx_use_flag_if_available(-Wno-delete-non-virtual-dtor)
  endif()

  # Check if our libraries have unresolved symbols
  #hpx_use_flag_if_available(-Wl,-z,defs NAME no_undefined_symbols)

  ##############################################################################
  # GCC-only configuration
  ##############################################################################
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(enable_visibility_hidden ON)
    if(040400 LESS ${GCC_VERSION})
      # visibility can't be disabled for gcc < 4.4
      set(enable_visibility_hidden OFF)
    elseif(${GCC_VERSION} GREATER 040699 AND ${GCC_VERSION} LESS 040800)
      # visibility is broken for gcc 4.7.x
      set(enable_visibility_hidden OFF)
    endif()

    if(${enable_visibility_hidden})
      option(HPX_ELF_HIDDEN_VISIBILITY
        "Use -fvisibility=hidden for Release and MinSizeRel builds (default: ON)" ON)

      if(HPX_ELF_HIDDEN_VISIBILITY)
        if("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel" OR
           "${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR
           "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")

          hpx_add_definitions(-DHPX_ELF_HIDDEN_VISIBILITY)
          hpx_add_definitions(-DHPX_COROUTINE_GCC_HAVE_VISIBILITY)
          hpx_add_definitions(-DHPX_PLUGIN_GCC_HAVE_VISIBILITY)
          hpx_use_flag_if_available(-fvisibility=hidden LANGUAGES CXX C Fortran)
        endif()
      endif()
    endif()
  endif()

  ##############################################################################
  # Clang-only configuration
  ##############################################################################
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    hpx_use_flag_if_available(-Wno-cast-align)
  endif()

  ##############################################################################
  # ICPC-only configuration
  ##############################################################################
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # Disable the following warnings:
    # #1170: invalid redeclaration of nested class
    # #858: type qualifier on return type is meaningless
    # #1098: the qualifier on this friend declaration is ignored
    # #488: template parameter not used in declaring the parameter type
    hpx_use_flag_if_available(-wd1170)
    hpx_use_flag_if_available(-wd858)
    hpx_use_flag_if_available(-wd1098)
    hpx_use_flag_if_available(-wd488)
    if("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel" OR
       "${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR
       "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
      ## FIXME: figure out which flags can be used with HPX
      #hpx_use_flag_if_available(-opt-calloc)
      #hpx_use_flag_if_available(-inline-calloc)
      #hpx_use_flag_if_available(-ip)
      #hpx_use_flag_if_available(-ipo)
      #hpx_use_flag_if_available(-opt-class-analysis)
      #hpx_use_flag_if_available(-opt-multi-version-aggressive)
      #hpx_use_flag_if_available(-xHost)
      #hpx_use_flag_if_available(-fno-alias)
      #hpx_use_flag_if_available(-fbuiltin)
      #hpx_use_flag_if_available(-finline)
      #hpx_use_flag_if_available(-fma)
      #hpx_use_flag_if_available(-fno-implicit-inline-templates)
      #hpx_use_flag_if_available(-fno-implicit-templates)
      #hpx_use_flag_if_available(-ftls-model=initial-exec)
    endif()
  endif()

  ##############################################################################
  # x86-64 specific configuration
  hpx_option(HPX_MICROACRH_OPTIMIZATIONS BOOL
             "Allow microarchitecture-specific optimizations (Default: OFF)"
             OFF ADVANCED)

  if(NOT HPX_MICROARCH_OPTIMIZATIONS)
    foreach(language CXX C Fortran)
      if(NOT HPX_${language}_FLAG_MARCH_NATIVE)
        set(HPX_${language}_FLAG_MARCH_NATIVE OFF)
      endif()

      if(NOT HPX_${language}_FLAG_MCX16)
        set(HPX_${language}_FLAG_MCX16 OFF)
      endif()

      if(NOT HPX_${language}_FLAG_MCX16)
        set(HPX_${language}_FLAG_MCX16 OFF)
      endif()
    endforeach()

    if(NOT HPX_SSE2)
      set(HPX_SSE2 OFF)
    endif()

    if(NOT HPX_RDTSCP)
      set(HPX_RDTSCP OFF)
    endif()
  endif()

  ##############################################################################
  # GCC's -march=native will automatically tune generated code for the host
  # environment. This is available on newish versions of GCC only (4.3ish). If
  # this flag is used, the generated binaries will be less portable.
  hpx_use_flag_if_available(-march=native LANGUAGES CXX C Fortran)

  # cmpxchg16b is an x86-64 extension present on most newer x86-64 machines.
  # It allows us to do double quadword (128bit) atomic compare and swap
  # operations, which is AWESOME. Note that early x86-64 processors do lack
  # this instruction.
  hpx_use_flag_if_available(-mcx16 LANGUAGES CXX C Fortran)

  if(HPX_CXX_FLAG_MCX16)
    hpx_add_definitions(-DHPX_HAVE_GNU_SYNC_16)
    hpx_add_definitions(-DBOOST_ATOMIC_HAVE_GNU_SYNC_16) # for the gnu code
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
    hpx_use_flag_if_available(-msse2 LANGUAGES CXX C Fortran)
  endif()

  # rdtsc is an x86 instruction that reads the value of a CPU time stamp
  # counter. rdtscp is an extension to rdtsc. The difference is that rdtscp is
  # a serializing instruction.
  hpx_cpuid("rdtsc" HPX_RDTSC
    DEFINITIONS HPX_HAVE_RDTSC)
  hpx_cpuid("rdtscp" HPX_RDTSCP
    DEFINITIONS HPX_HAVE_RDTSCP)

  hpx_option(HPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION BOOL
      "Disable automatic registration with Boost.Serialization (Default: Off)" Off ADVANCED)

  ##############################################################################
  # Macro definitions for system headers
  ##############################################################################
  hpx_add_definitions(-D_GNU_SOURCE)

  ##############################################################################
  # System libraries
  ##############################################################################
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
    set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} dl)
  endif()

  if(NOT APPLE)
    set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} rt)
  endif()

  hpx_check_for_pthread_affinity_np(HPX_PTHREAD_AFFINITY_NP
    DEFINITIONS HPX_HAVE_PTHREAD_AFFINITY_NP)

  hpx_use_flag_if_available(-pthread LANGUAGES CXX C)

endif()
  
if(HPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION)
  hpx_add_definitions(-DHPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION)
endif()

################################################################################
# target specification
################################################################################
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

###############################################################################
# Activate plugins
###############################################################################
add_plugin_modules()

###############################################################################
hpx_option(HPX_BUILD_TESTS BOOL "Build HPX tests (default: ON)" ON ADVANCED)

if(HPX_BUILD_TESTS)
  hpx_option(HPX_BUILD_TESTS_BENCHMARKS BOOL "Build HPX benchmark tests (default: ON)" ON ADVANCED)
  hpx_option(HPX_BUILD_TESTS_REGRESSIONS BOOL "Build HPX regression tests (default: ON)" ON ADVANCED)
  hpx_option(HPX_BUILD_TESTS_UNIT BOOL "Build HPX unit tests (default: ON)" ON ADVANCED)

  add_hpx_pseudo_target(tests)

  add_custom_command(TARGET tests POST_BUILD
    COMMAND ${CMAKE_SOURCE_DIR}/python/scripts/hpx_run_tests.py
    ARGS --prefix=${CMAKE_BINARY_DIR}/bin/
         --args=\${HPX_TEST_ARGUMENTS}
         --log-stdout
         --no-exit-code
         ${CMAKE_BINARY_DIR}/share/hpx-${HPX_VERSION}/tests/hpx_unit.tests
         ${CMAKE_BINARY_DIR}/share/hpx-${HPX_VERSION}/tests/hpx_regressions.tests
    COMMENT "Running HPX tests")

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

if(${BOOST_MINOR_VERSION} GREATER 50)
  add_subdirectory(preprocess)
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
    PATTERN ".git" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "CTestFiles" EXCLUDE)

  install( # install all hpx header files
    DIRECTORY hpx
    DESTINATION share/hpx-${HPX_VERSION}/doc/html/code
    COMPONENT core
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".git" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "CTestFiles" EXCLUDE)

  install( # install all hpx source files
    DIRECTORY src
    DESTINATION share/hpx-${HPX_VERSION}/doc/html/code
    COMPONENT core
    PATTERN ".git" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "CTestFiles" EXCLUDE)

  install( # install all hpx examples source files
    DIRECTORY examples
    DESTINATION share/hpx-${HPX_VERSION}/doc/html/code
    COMPONENT core
    PATTERN ".git" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "CTestFiles" EXCLUDE)

  install( # install all hpx tests source files
    DIRECTORY tests
    DESTINATION share/hpx-${HPX_VERSION}/doc/html/code
    COMPONENT core
    PATTERN ".git" EXCLUDE
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "CTestFiles" EXCLUDE)

  install( # install all hpx cmake utility files
    DIRECTORY cmake/
    DESTINATION share/hpx-${HPX_VERSION}/cmake
    COMPONENT core
    PATTERN ".cmake.in" EXCLUDE
    PATTERN ".git" EXCLUDE)

  install( # install hpx python scripts
    DIRECTORY python/scripts/
    DESTINATION bin
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                     GROUP_READ GROUP_EXECUTE
                     WORLD_READ WORLD_EXECUTE
    COMPONENT core
    FILES_MATCHING PATTERN "*.py"
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
    PATTERN ".git" EXCLUDE)

  # create a symlink in share pointing to the latest HPX installation
  install(CODE
    "set(sharedir \"${CMAKE_INSTALL_PREFIX}/share/\")
     execute_process(
     COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
             \"hpx-${HPX_VERSION}\" \"hpx\"
     WORKING_DIRECTORY \"\${sharedir}\")")

  install( # install external dependencies
    DIRECTORY external/cache/boost
              external/endian/boost
    DESTINATION include/hpx/external
    COMPONENT core
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".git" EXCLUDE)

  if(${BOOST_MINOR_VERSION} LESS 49)
    # install Boost.Serialization patch for older Boost versions only
    install(
      DIRECTORY external/serialization/boost
      DESTINATION include/hpx/external
      COMPONENT core
      FILES_MATCHING PATTERN "*.hpp"
      PATTERN ".git" EXCLUDE)
  endif()

  if(${BOOST_MINOR_VERSION} LESS 53)
    # install Boost.Atomic library for older Boost versions only
    install(
      DIRECTORY external/atomic/boost
      DESTINATION include/hpx/external
      COMPONENT core
      FILES_MATCHING PATTERN "*.hpp"
      PATTERN ".git" EXCLUDE)
  endif()

  if(${BOOST_MINOR_VERSION} LESS 53)
    # install Boost.Lockfree library for older Boost versions only
    install(
      DIRECTORY external/lockfree/boost
      DESTINATION include/hpx/external
      COMPONENT core
      FILES_MATCHING PATTERN "*.hpp"
      PATTERN ".git" EXCLUDE)
  endif()

  if(HPX_INTERNAL_MOVE)
    # Install Boost.Move if requested
    install(
      DIRECTORY external/move/boost
      DESTINATION include
      COMPONENT core
      FILES_MATCHING PATTERN "*.hpp"
      PATTERN ".git" EXCLUDE)
  endif()

  if(HPX_INTERNAL_CHRONO)
    install( # install Boost.Chrono headers if we're using our internal version
      DIRECTORY external/chrono/boost
      DESTINATION include/hpx/external
      COMPONENT core
      FILES_MATCHING PATTERN "*.hpp"
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
      PATTERN ".git" EXCLUDE)

    install(
      DIRECTORY ${hpx_SOURCE_DIR}/docs/html/images/
      DESTINATION share/hpx-${HPX_VERSION}/doc/html/images
      COMPONENT docs
      FILES_MATCHING PATTERN "*.png"
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
set(HPX_INCLUDE_DIRECTORIES
    ${CMAKE_INSTALL_PREFIX}/include/hpx/external ${HPX_INCLUDE_DIRECTORIES})

list(REMOVE_DUPLICATES HPX_LINK_DIRECTORIES)
list(REMOVE_DUPLICATES HPX_INCLUDE_DIRECTORIES)

set(external_rpath "")
set(external_link_directories "")
set(external_libraries "")
set(external_libraries_name "")
set(external_link_flags "")
set(external_include_directories "")
set(external_include_flags "")

foreach(directory ${HPX_LINK_DIRECTORIES})
  string(REGEX REPLACE "([ ()])" "\\\\\\1" directory "${directory}")
  set(external_link_directories "${external_link_directories} ${directory}")
  set(external_link_flags "${external_link_flags} -L${directory}")
  set(external_rpath "${external_rpath}:${directory}")
endforeach()

foreach(lib ${hpx_LIBRARIES})
  get_filename_component(lib ${lib} NAME_WE)
  string(REGEX REPLACE "^lib" "" lib ${lib})
  set(external_libraries "${external_libraries} ${lib}")
  set(external_libraries_name "${external_libraries_name} -l${lib}")
endforeach()

foreach(directory ${HPX_INCLUDE_DIRECTORIES})
  string(REGEX REPLACE "([ ()])" "\\\\\\1" directory "${directory}")
  set(external_include_directories "${external_include_directories} ${directory}")
  set(external_include_flags "${external_include_flags} -I${directory}")
endforeach()

set(boost_library_info "")

foreach(library ${BOOST_LIBRARIES})
  string(TOUPPER ${library} library_uc)
  string(REGEX REPLACE "([ ()])" "\\\\\\1" "BOOST_${library_uc}_LIBRARY" "${BOOST_${library_uc}_LIBRARY}")
  set(boost_library_info
    "${boost_library_info}set(HPX_BOOST_${library_uc}_LIBRARY ${BOOST_${library_uc}_LIBRARY}\n")
  set(boost_library_info
    "${boost_library_info}    CACHE FILEPATH \"Boost ${library} shared library.\")\n")
endforeach()

get_directory_property(definitions DEFINITIONS)

set(external_definitions "")

foreach(definition ${definitions})
    string(REGEX REPLACE "\"" "\\\\\\\\\\\\\\\\\"" definition ${definition})
    set(external_definitions "${external_definitions} ${definition}")
endforeach()

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

set(hpx_link_flag_target_properties "")
if(MSVC AND HPX_LINK_FLAG_TARGET_PROPERTIES)
  set(hpx_link_flag_target_properties "${HPX_LINK_FLAG_TARGET_PROPERTIES}")
endif()

configure_file(${hpx_SOURCE_DIR}/cmake/templates/FindHPX.cmake.in
               ${output_dir}/share/${cmake_dir}/Modules/FindHPX.cmake
               ESCAPE_QUOTES @ONLY)

if(NOT HPX_NO_INSTALL)
  install(FILES ${output_dir}/share/${cmake_dir}/Modules/FindHPX.cmake
          DESTINATION share/${cmake_dir}/Modules)
endif()

if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
    configure_file(${hpx_SOURCE_DIR}/cmake/templates/hpx_application.pc.in
                   ${output_dir}/lib/pkgconfig/hpx_application.pc
                   @ONLY)
    configure_file(${hpx_SOURCE_DIR}/cmake/templates/hpx_component.pc.in
                   ${output_dir}/lib/pkgconfig/hpx_component.pc
                   @ONLY)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    configure_file(${hpx_SOURCE_DIR}/cmake/templates/hpx_application_debug.pc.in
                   ${output_dir}/lib/pkgconfig/hpx_application_debug.pc
                   @ONLY)

    configure_file(${hpx_SOURCE_DIR}/cmake/templates/hpx_component_debug.pc.in
                   ${output_dir}/lib/pkgconfig/hpx_component_debug.pc
                   @ONLY)
endif()

if(NOT HPX_NO_INSTALL)
  if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
    install(FILES ${output_dir}/lib/pkgconfig/hpx_application.pc
            DESTINATION lib/pkgconfig)
    install(FILES ${output_dir}/lib/pkgconfig/hpx_component.pc
            DESTINATION lib/pkgconfig)
  endif()

  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    install(FILES ${output_dir}/lib/pkgconfig/hpx_application_debug.pc
            DESTINATION lib/pkgconfig)
    install(FILES ${output_dir}/lib/pkgconfig/hpx_component_debug.pc
            DESTINATION lib/pkgconfig)
  endif()
endif()

file(MAKE_DIRECTORY ${output_dir}/share/hpx-${HPX_VERSION}/tests)

# create a symlink in share pointing to the latest HPX installation
execute_process(COMMAND
  "${CMAKE_COMMAND}" -E create_symlink "hpx-${HPX_VERSION}" "hpx"
  WORKING_DIRECTORY ${output_dir}/share)

file(WRITE ${output_dir}/share/hpx-${HPX_VERSION}/tests/hpx_unit.tests
           "[\n${HPX_UNIT_TEST_LIST}\n]\n")

file(WRITE ${output_dir}/share/hpx-${HPX_VERSION}/tests/hpx_regressions.tests
           "[\n${HPX_REGRESSION_TEST_LIST}\n]\n")

if(NOT HPX_NO_INSTALL)
  install(FILES ${output_dir}/share/hpx-${HPX_VERSION}/tests/hpx_unit.tests
          DESTINATION share/hpx-${HPX_VERSION}/tests)
  install(FILES ${output_dir}/share/hpx-${HPX_VERSION}/tests/hpx_regressions.tests
          DESTINATION share/hpx-${HPX_VERSION}/tests)
endif()

unset(HPX_UNIT_TEST_LIST CACHE)
unset(HPX_REGRESSION_TEST_LIST CACHE)


