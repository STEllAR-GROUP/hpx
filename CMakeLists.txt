# Copyright (c) 2007-2014 Hartmut Kaiser
# Copyright (c) 2011-2013 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
# Copyright (c)      2013 Jeroen Habraken
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least CMake V2.8.10
cmake_minimum_required(VERSION 2.8.10 FATAL_ERROR)

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
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)")
endif()

set(HPX_BUILD_TYPE "${CMAKE_BUILD_TYPE}")

################################################################################
# project metadata
################################################################################
project(hpx CXX C)

set(HPX_MAJOR_VERSION 0)
set(HPX_MINOR_VERSION 9)
set(HPX_PATCH_LEVEL   9)
set(HPX_VERSION "${HPX_MAJOR_VERSION}.${HPX_MINOR_VERSION}.${HPX_PATCH_LEVEL}")
set(HPX_SOVERSION ${HPX_MAJOR_VERSION})

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

################################################################################
# CMake configuration
################################################################################
set(CMAKE_MODULE_PATH "${hpx_SOURCE_DIR}/cmake")

include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

# include additional macro definitions
include(HPX_Utils)

set(HPX_DEFINITIONS CACHE INTERNAL "" FORCE)
set(HPX_CONFIG_DEFINITIONS CACHE INTERNAL "" FORCE)
set(HPX_LIBRARIES CACHE INTERNAL FORCE)

hpx_force_out_of_tree_build("This project requires an out-of-source-tree build. See README.rst. Clean your CMake cache and CMakeFiles if this message persists.")

if(NOT HPX_CMAKE_LOGLEVEL)
  set(HPX_CMAKE_LOGLEVEL "WARN")
endif()

################################################################################
# Fortran compiler detection
#
include(HPX_FortranCompiler)
################################################################################

################################################################################
# Setup platform for which HPX should be compiled for.
#
include(HPX_SetPlatform)
################################################################################

################################################################################
# Set our build options cache variables which are customizable by users
#

## Generic build options
set(DEFAULT_MALLOC "system")
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  if(HPX_PLATFORM_UC STREQUAL "XEONPHI")
    set(DEFAULT_MALLOC "tbbmalloc")
  else()
    set(DEFAULT_MALLOC "tcmalloc")
#    set(DEFAULT_MALLOC "jemalloc")
  endif()
endif()

hpx_option(WITH_MALLOC
  STRING
  "Define which allocator should be linked in. Options are: system, tcmalloc, jemalloc and tbbmalloc (default is: tcmalloc)"
  ${DEFAULT_MALLOC} ADVANCED)

set(WITH_HWLOC_DEFAULT ON)
if(APPLE)
  set(WITH_HWLOC_DEFAULT OFF)
endif()

hpx_option(WITH_HWLOC
  BOOL
  "Use Hwloc for hardware topolgy information and thread pinning. If disabled, performance might be reduced."
  ${WITH_HWLOC_DEFAULT} ADVANCED)

hpx_option(WITH_SECURITY BOOL "Enable security support via libsodium." OFF)

# Logging configuration
hpx_option(WITH_LOGGING BOOL "Build HPX with logging enabled (default: ON)." ON ADVANCED)
if(NOT WITH_LOGGING)
  hpx_add_config_define(HPX_NO_LOGGING)
endif()

## Compiler related build options
hpx_option(WITH_GCC_VERSION_CHECK BOOL "Ignore version reported by gcc (default: ON)" ON ADVANCED)

hpx_option(WITH_DEFAULT_BUILD_TARGETS BOOL
  "Associate the core HPX library with the default build target (default: ON)." ON ADVANCED)

hpx_option(WITH_WARNINGS BOOL "Enable compiler warnings (default: ON)" ON ADVANCED)

################################################################################
# Native TLS configuration
################################################################################
set(WITH_NATIVE_TLS_DEFAULT ON)
if(APPLE)
  set(WITH_NATIVE_TLS_DEFAULT OFF)
endif()
hpx_option(WITH_NATIVE_TLS BOOL "Use native TLS support if available (default: ON)" ${WITH_NATIVE_TLS_DEFAULT} ADVANCED)
if(WITH_NATIVE_TLS)
  hpx_info("Native TLS is enabled.")
  hpx_add_config_define(HPX_HAVE_NATIVE_TLS)
endif()

################################################################################
# Disable the use of partially preprocessed header files
################################################################################
hpx_option(WITH_PREPROCESSOR_LIMIT_EXPANSION BOOL "Do not use preprocessed headers (default: OFF)" OFF ADVANCED)

if(WITH_PREPROCESSOR_LIMIT_EXPANSION)
  hpx_info("Use of preprocessed headers is disabled.")
  hpx_add_config_define(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
endif()

set(WITH_HIDDEN_VISIBILITY_DEFAULT ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.6.99
      AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
    set(WITH_HIDDEN_VISIBILITY_DEFAULT OFF)
    message("....")
  endif()
  if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
    set(WITH_HIDDEN_VISIBILITY_DEFAULT OFF)
  endif()
endif()

hpx_option(WITH_HIDDEN_VISIBILITY BOOL
  "Use -fvisibility=hidden for builds on platforms which support it (default ON)"
  ${WITH_HIDDEN_VISIBILITY_DEFAULT}
  ADVANCED)

## Thread Manager related build options

set(WITH_MAX_CPU_COUNT_DEFAULT "64")
if(HPX_PLATFORM_UC STREQUAL "XEONPHI")
  set(WITH_MAX_CPU_COUNT_DEFAULT "256")
endif()
hpx_option(WITH_MAX_CPU_COUNT STRING
  "HPX applications will not use more that this number of OS-Threads (default: ${WITH_MAX_CPU_COUNT_DEFAULT})"
  ${WITH_MAX_CPU_COUNT_DEFAULT}
  CATEGORY "Thread Manager" ADVANCED)

hpx_option(WITH_THREAD_STACK_MMAP BOOL
  "Use mmap for stack allocation on appropriate platforms"
  ON
  CATEGORY "Thread Manager" ADVANCED)

hpx_option(WITH_THREAD_MANAGER_IDLE_BACKOFF BOOL
  "Use mmap for stack allocation on appropriate platforms (default: OFF)"
  OFF
  CATEGORY "Thread Manager" ADVANCED)

hpx_option(WITH_STACKTRACES BOOL "Attach backtraces to HPX exceptions (default: ON)"
  ON CATEGORY "Thread Manager" ADVANCED)

if(WITH_STACKTRACES OR WITH_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION)
  hpx_info("Stack traces are enabled.")
  hpx_add_config_define(HPX_HAVE_STACKTRACES)
endif()

hpx_option(WITH_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION BOOL
  "Enable thread stack back trace being captured on suspension (default: OFF)"
  OFF
  CATEGORY "Thread Manager" ADVANCED)

hpx_option(WITH_THREAD_BACKTRACE_ON_SUSPENSION_DEPTH STRING
  "Thread stack back trace depth being captured on suspension (default: 5)"
  "5" CATEGORY "Thread Manager" ADVANCED)

hpx_option(WITH_THREAD_MAINTAIN_FULLBACKTRACE_ON_SUSPENSION BOOL
  "Enable thread stack back trace being captured on suspension (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

if(WITH_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION)
  hpx_add_config_define(HPX_THREAD_MAINTAIN_BACKTRACE_ON_SUSPENSION)
  hpx_add_config_define(HPX_THREAD_BACKTRACE_ON_SUSPENSION_DEPTH
    ${WITH_THREAD_BACKTRACE_ON_SUSPENSION_DEPTH})
  if(WITH_THREAD_MAINTAIN_FULLBACKTRACE_ON_SUSPENSION)
    hpx_add_config_define(HPX_THREAD_MAINTAIN_FULLBAKCTRACE_ON_SUSPENSION))
  endif()
endif()

hpx_option(WITH_THREAD_MAINTAIN_TARGET_ADDRESS BOOL
  "Enable storing target address in thread for NUMA awareness (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

if(WITH_THREAD_MAINTAIN_TARGET_ADDRESS)
  hpx_add_config_define(HPX_THREAD_MAINTAIN_TARGET_ADDRESS)
endif()

hpx_option(WITH_THREAD_MAINTAIN_QUEUE_WAITTIME BOOL
  "Enable collecting queue wait times for threads (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

if(WITH_THREAD_MAINTAIN_QUEUE_WAITTIME)
  hpx_add_config_define(HPX_THREAD_MAINTAIN_QUEUE_WAITTIME)
endif()

hpx_option(WITH_THREAD_MAINTAIN_IDLE_RATES BOOL
  "Enable measuring the percentage of overhead times spent in the scheduler (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

hpx_option(WITH_THREAD_MAINTAIN_CREATION_AND_CLEANUP_RATES BOOL
  "Enable measuring thread creation and cleanup times (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

if(WITH_THREAD_MAINTAIN_IDLE_RATES)
  hpx_add_config_define(HPX_THREAD_MAINTAIN_IDLE_RATES)
  if(WITH_THREAD_MAINTAIN_CREATION_AND_CLEANUP_RATES)
    hpx_add_config_define(HPX_THREAD_MAINTAIN_CREATION_AND_CLEANUP_RATES)
  endif()
endif()

hpx_option(WITH_THREAD_MAINTAIN_CUMULATIVE_COUNTS BOOL
  "Enable keeping track of cumulative thread counts in the schedulers (default: ON)"
  ON CATEGORY "Thread Manager" ADVANCED)

if(WITH_THREAD_MAINTAIN_CUMULATIVE_COUNTS)
  hpx_add_config_define(HPX_THREAD_MAINTAIN_CUMULATIVE_COUNTS)
endif()

hpx_option(WITH_THREAD_MAINTAIN_STEALING_COUNTS BOOL
  "Enable keeping track of counts of thread stealing incidents in the schedulers (default: ON)"
  ON CATEGORY "Thread Manager" ADVANCED)

if(WITH_THREAD_MAINTAIN_STEALING_COUNTS)
  hpx_add_config_define(HPX_THREAD_MAINTAIN_STEALING_COUNTS)
endif()

hpx_option(WITH_THREAD_MAINTAIN_LOCAL_STORAGE BOOL
  "Enable thread local storage for all HPX threads (default: ON)"
  ON CATEGORY "Thread Manager" ADVANCED)

if(WITH_THREAD_MAINTAIN_LOCAL_STORAGE)
  hpx_add_config_define(HPX_THREAD_MAINTAIN_LOCAL_STORAGE)
endif()

hpx_option(WITH_SWAP_CONTEXT_EMULATION BOOL "Emulate SwapContext API for coroutines (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

hpx_option(WITH_THREAD_SCHEDULERS STRING
  "Which thread schedulers are build. Options are: all, local, static-priority, hierarchy, and periodic-priority. For multiple enabled schedulers, separate with a semicolon (default: ALL)"
  "all"
  CATEGORY "Thread Manager" ADVANCED)

## AGAS related build options
hpx_option(WITH_AGAS_DUMP_REFCNT_ENTRIES BOOL
  "Enable dumps of the AGAS refcnt tables to logs (default: OFF)"
  OFF CATEGORY "AGAS" ADVANCED)
if(WITH_AGAS_DUMP_REFCNT_ENTRIES)
  hpx_add_config_define(HPX_AGAS_DUMP_REFCNT_ENTRIES)
endif()

## Parcelport related build options
set(WITH_PARCELPORT_TCP_DEFAULT ON)
if(HPX_PLATFORM_UC STREQUAL "BLUEGENEQ")
  set(WITH_PARCELPORT_TCP_DEFAULT OFF)
endif()

hpx_option(WITH_PARCELPORT_IBVERBS BOOL "Enable the ibverbs based parcelport. This is currently an experimental feature" OFF CATEGORY "Parcelport" ADVANCED)
hpx_option(WITH_PARCELPORT_IPC BOOL "Enable the IPC (inter process communication) based parcelport. This is currently an experimental feature" OFF CATEGORY "Parcelport" ADVANCED)
hpx_option(WITH_PARCELPORT_MPI BOOL "Enable the MPI based parcelport." ON CATEGORY "Parcelport")
hpx_option(WITH_PARCELPORT_TCP BOOL "Enable the TCP based parcelport." ${WITH_PARCELPORT_TCP_DEFAULT} CATEGORY "Parcelport")

## ibverbs parcelport settings
hpx_option(WITH_PARCELPORT_IBVERBS_IFNAME STRING "The interface name of the ibverbs capable network adapter (default: ib0)" "ib0" CATEGORY "Parcelport" ADVANCED)
hpx_option(WITH_PARCELPORT_IBVERBS_MESSAGE_PAYLOAD STRING "Size of the message payload not sent with RDMA (default: 512 byte)" "512" CATEGORY "Parcelport" ADVANCED)
hpx_option(WITH_PARCELPORT_IBVERBS_MEMORY_CHUNK_SIZE STRING "Number of bytes a chunk in the memory pool can hold (default: 64MB)" "67108864" CATEGORY "Parcelport" ADVANCED)
hpx_option(WITH_PARCELPORT_IBVERBS_MAX_MEMORY_CHUNKS STRING "Maximum number of chunks that can be allocated (default: 100)" "100" CATEGORY "Parcelport" ADVANCED)

## mpi parcelport settings
hpx_option(WITH_PARCELPORT_MPI_ENV STRING
  "List of environment variables checked to detect MPI (default: PMI_RANK;OMPI_COMM_WORLD_SIZE)."
  "PMI_RANK;OMPI_COMM_WORLD_SIZE" CATEGORY "Parcelport" ADVANCED)

## Profiling related build options
hpx_option(WITH_APEX BOOL "Enable APEX instrumentation support." OFF CATEGORY "Profiling")
hpx_option(WITH_PAPI BOOL "Enable the PAPI based performance counter." OFF CATEGORY "Profiling")
hpx_option(WITH_TAU BOOL "Enable TAU profiling support." OFF CATEGORY "Profiling")
hpx_option(WITH_GOOGLE_PERFTOOLS BOOL "Enable Google Perftools instrumentation support." OFF CATEGORY "Profiling")
hpx_option(WITH_ITTNOTIFY BOOL "Enable Amplifier (ITT) instrumentation support." OFF CATEGORY "Profiling")

## Debugging related build options
hpx_option(WITH_VALGRIND BOOL "Enable Valgrind instrumentation support." OFF CATEGORY "Debugging")

set(WITH_VERIFY_LOCKS_DEFAULT OFF)
set(WITH_VERIFY_LOCKS_GLOBALLY_DEFAULT OFF)
set(WITH_THREAD_DEBUG_INFO_DEFAULT OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(WITH_VERIFY_LOCKS_DEFAULT ON)
  set(WITH_VERIFY_LOCKS_GLOBALLY_DEFAULT ON)
  set(WITH_THREAD_DEBUG_INFO_DEFAULT ON)
endif()
hpx_option(WITH_VERIFY_LOCKS BOOL
  "Enable lock verification code (default: ${WITH_VERIFY_LOCKS_DEFAULT})"
  ${WITH_VERIFY_LOCKS_DEFAULT}
  CATEGORY "Debugging" ADVANCED)
hpx_option(WITH_VERIFY_LOCKS_GLOBALLY BOOL
  "Enable global lock verification code (default: ${WITH_VERIFY_LOCKS_GLOBALLY_DEFAULT})"
  ${WITH_VERIFY_LOCKS_GLOBALLY_DEFAULT}
  CATEGORY "Debugging" ADVANCED)
hpx_option(WITH_VERIFY_LOCKS_BACKTRACE BOOL
  "Enable thred stack back trace being captured on lock registration (to be used in combination with WITH_VERIFY_LOCKS=ON, default: OFF)"
  OFF
  CATEGORY "Debugging" ADVANCED)
hpx_option(WITH_THREAD_DEBUG_INFO BOOL
  "Enable thread debugging information (default: ${WITH_THREAD_DEBUG_INFO_DEFAULT})"
  ${WITH_THREAD_DEBUG_INFO_DEFAULT}
  CATEGORY "Debugging" ADVANCED)
hpx_option(WITH_THREAD_GUARD_PAGE BOOL
  "Enable thread guard page (default: ON)"
  ON
  CATEGORY "Debugging" ADVANCED)

if(WITH_VERIFY_LOCKS)
  hpx_add_config_define(HPX_HAVE_VERIFY_LOCKS)
  if(WITH_VERIFY_LOCKS_BACKTRACE)
    hpx_add_config_define(HPX_HAVE_VERIFY_LOCKS_BACKTRACE)
  endif()
endif()
if(WITH_VERIFY_LOCKS_GLOBALLY)
  hpx_add_config_define(HPX_HAVE_VERIFY_LOCKS_GLOBALLY)
endif()

# Additional debug support
if(NOT MSVC AND WITH_THREAD_GUARD_PAGE)
  hpx_add_config_define(HPX_THREAD_GUARD_PAGE)
endif()

if(NOT MSVC AND WITH_THREAD_STACK_MMAP)
  hpx_add_config_define(HPX_USE_MMAP)
endif()

if(WITH_THREAD_MANAGER_IDLE_BACKOFF)
  hpx_add_config_define(HPX_THREAD_BACKOFF_ON_IDLE)
endif()

if(WITH_THREAD_DEBUG_INFO)
  hpx_add_config_define(HPX_THREAD_MAINTAIN_PARENT_REFERENCE)
  hpx_add_config_define(HPX_THREAD_MAINTAIN_PHASE_INFORMATION)
  hpx_add_config_define(HPX_THREAD_MAINTAIN_DESCRIPTION)
  hpx_add_config_define(HPX_THREAD_MAINTAIN_DEADLOCK_DETECTION)
endif()

################################################################################

##############################################################################
# Find Our dependencies:
#   These are all dependencies needed to build the core library. Dependencies
#   that are only needed by plugins, examples or tests should be found
#   seperately in the appropriate subdirectory.
# When a package is found, the necessary include paths are set, and the libraries
# are added to the HPX_LIBRARIES variables

# Setting up our required Boost libraries. This will find all required boost
# libraries and add possibly needed boost headers shipped with the hpx source
include(HPX_SetupBoost)

# Find all allocators which are currently supported.
include(HPX_SetupAllocator)

if(WITH_HWLOC)
  find_package(Hwloc)
  if(NOT HWLOC_FOUND)
    hpx_error("Hwloc could not be found and WITH_HWLOC=ON, please specify HWLOC_ROOT to point to the correct location or set WITH_HWLOC to OFF")
  endif()
  hpx_libraries(${HWLOC_LIBRARIES})
  include_directories(${HWLOC_INCLUDE_DIR})
  add_config_define(HPX_HAVE_HWLOC)
endif()

################################################################################
# Decide whether to use the ibverbs based parcelport
################################################################################
if(WITH_PARCELPORT_IBVERBS)
  find_package(Ibverbs)
  if(NOT IBVERBS_FOUND)
    hpx_error("The ibverbs libraries could not be found and WITH_PARCELPORT_IBVERBS=ON, please specify IBVERBS_ROOT to point to the correct location or set WITH_PARCELPORT_IBVERBS to OFF")
  endif()
  hpx_libraries(${IBVERBS_LIBRARIES})
  include_directories(${IBVERBS_INCLUDE_DIR})

  find_package(Rdmacm)
  if(NOT RDMACM_FOUND)
    hpx_error("The ibverbs libraries could not be found and WITH_PARCELPORT_IBVERBS=ON, please specify RDMACM_ROOT to point to the correct location or set WITH_PARCELPORT_IBVERBS to OFF")
  endif()
  hpx_libraries(${RDMACM_LIBRARIES})
  include_directories(${RDMACM_INCLUDE_DIR})
  hpx_add_config_define(HPX_HAVE_PARCELPORT_IBVERBS)
  hpx_add_config_define(HPX_PARCELPORT_IBVERBS_IFNAME "\"${WITH_PARCELPORT_IBVERBS_IFNAME}\"")
  hpx_add_config_define(HPX_PARCELPORT_IBVERBS_MESSAGE_PAYLOAD "${WITH_PARCELPORT_IBVERBS_MESSAGE_PAYLOAD}")
  hpx_add_config_define(HPX_PARCELPORT_IBVERBS_MEMORY_CHUNK_SIZE "${WITH_PARCELPORT_IBVERBS_MEMORY_CHUNK_SIZE}")
  hpx_add_config_define(HPX_PARCELPORT_IBVERBS_MAX_MEMORY_CHUNKS "${WITH_PARCELPORT_IBVERBS_MAX_MEMORY_CHUNKS}")
endif()

################################################################################
# Decide whether to use shared memory based parcelport (starting with
# Boost V1.52)
################################################################################
if(WITH_PARCELPORT_IPC)
  if(Boost_VERSION GREATER 105100)
    hpx_add_config_define(HPX_HAVE_PARCELPORT_IPC)
  endif()
endif()

################################################################################
# Decide whether to use the MPI based parcelport
################################################################################
if(WITH_PARCELPORT_MPI)
  find_package(MPI)
  if(NOT MPI_CXX_FOUND)
    hpx_error("MPI could not be found and WITH_PARCELPORT_MPI=On, please specify MPI_CXX_COMPILER to point to a working MPI C++ compiler for your platform")
  endif()
  hpx_add_config_define(HPX_HAVE_PARCELPORT_MPI)
  if(MPI_CXX_COMPILE_FLAGS)
    hpx_add_compile_flag(${MPI_CXX_COMPILE_FLAGS})
  endif()
  if(MPI_CXX_INCLUDE_PATH)
    include_directories(${MPI_CXX_INCLUDE_PATH})
  endif()
  if(MPI_CXX_LIBRARIES)
    hpx_libraries(${MPI_CXX_LIBRARIES})
  endif()
  if(MPI_LIBRARY)
    hpx_libraries(${MPI_LIBRARY})
  endif()
  if(MPI_EXTRA_LIBRARY)
    hpx_libraries(${MPI_LIBRARY})
  endif()
  # This list is to detect whether we run inside an mpi environment.
  # If one of those environment variables is set, the MPI parcelport
  # is enabled by default.
  # PMI_RANK: Intel MPI and MVAPICH2
  # OMPI_COMM_WORLD_SIZE: OpenMPI starting at version 1.3
  if(WITH_PARCELPORT_MPI_ENV)
    string(REPLACE ";" "," hpx_parcelport_mpi_env_ "${WITH_PARCELPORT_MPI_ENV}")
    hpx_add_config_define(HPX_PARCELPORT_MPI_ENV "\"${hpx_parcelport_mpi_env_}\"")
  endif()
endif()

################################################################################
# Decide whether to use the TCP/IP based parcelport
################################################################################
if(WITH_PARCELPORT_TCP)
  hpx_add_config_define(HPX_HAVE_PARCELPORT_TCP)
endif()

################################################################################
# Enable integration with Intel Amplifier and Inspector tools
################################################################################
if(WITH_AMPLIFIER)
  find_package(Amplifier)
  if(NOT AMPLIFIER_FOUND)
    hpx_error("Intel Amplifier could not be found and WITH_AMPLIFIER=On, please specify AMPLIFIER_ROOT to point to the root of your Amplifier installation")
  endif()
  hpx_libraries(${AMPLIFIER_LIBRARIES})
  include_directories(${AMPLIFIER_INCLUDE_DIR})
  hpx_add_config_define(HPX_HAVE_ITTNOTIFY)
endif()

if(WITH_TAU)
  find_package(TAU)
  if(NOT TAU_FOUND)
    hpx_error("TAU could not be found and WITH_TAU=On, please specify TAU_ROOT to point to the root of your TAU installation")
  endif()
  hpx_libraries(${TAU_LIBRARIES})
  include_directories(${TAU_INCLUDE_DIR})
endif()

################################################################################
# Enable integration with Apex event counters
################################################################################
if(WITH_APEX)
  find_package(APEX)
  if(NOT APEX_FOUND)
    hpx_error("Apex could not be found and WITH_APEX=On, please specify APEX_ROOT to point to the root of your Apex installation")
  endif()
  hpx_libraries(${APEX_LIBRARIES})
  include_directories(${APEX_INCLUDE_DIR})
  if(AMPLIFIER_FOUND)
    hpx_error("AMPLIFIER_FOUND has been set. Please disable the use of the Intel Amplifier (WITH_AMPLIFIER=Off) in order to use Apex")
  endif()
  hpx_add_config_define(HPX_HAVE_ITTNOTIFY)
endif()

if(WITH_GOOGLE_PERFTOOLS)
  find_package(GooglePerftools)
  if(NOT GOOGLE_PERFTOOLS_FOUND)
    hpx_error("Google Perftools could not be found and WITH_GOOGLE_PERFTOOLS=On, please specify GOOGLE_PERFTOOLS to point to the root of your Google Perftools installation")
  endif()
  hpx_libraries(${GOOGLE_PERFTOOLS_LIBRARIES})
  include_directories(${GOOGLE_PERFTOOLS_INCLUDE_DIR})
endif()

if(WITH_VALGRIND)
  find_package(Valgrind)
  if(NOT VALGRIND_FOUND)
    hpx_error("Valgrind could not be found and WITH_VALGRIND=On, please specify VALGRIND_ROOT to point to the root of your Valgrind installation")
  endif()
  include_directories(${VALGRIND_INCLUDE_DIR})
  hpx_add_config_define(HPX_HAVE_VALGRIND)
endif()
################################################################################

##############################################################################
# Check for compiler compatibility
#

# Check if the selected compiler versions are supposed to work with our codebase
if(CMAKE_COMPILER_IS_GNUCXX AND WITH_GCC_VERSION_CHECK)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.4.4)
    hpx_error("GCC 4.4.5 or higher is required. Specify WITH_GCC_VERSION_CHECK=OFF to ignore this error.")
  endif()
endif()

if(MSVC)
  if(NOT (MSVC11 OR MSVC12))
    hpx_error("MSVC x64 2012 or higher is required.")
  elseif(NOT CMAKE_CL_64)
    hpx_warn("MSVC (32Bit) will compile but will fail running larger applications because of limitations in the Windows OS.")
  endif()
endif()

# Setup platform specific compiler options and check for compatible compilers
if(HPX_PLATFORM_UC STREQUAL "NATIVE")
  hpx_info("Compiling with the native toolset")
elseif(HPX_PLATFORM_UC STREQUAL "ANDROID")
  hpx_info("Compiling for Android devices")
elseif(HPX_PLATFORM_UC STREQUAL "XEONPHI")
  hpx_info("Compiling for Intel Xeon Phi devices")
  if(NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
    hpx_error("HPX on the MIC can only be compiled with the Intel compiler.")
  endif()
  hpx_add_config_define(HPX_NATIVE_MIC)
  # We need to append the -mmic compile flag here.
  hpx_add_compile_flag(-mmic)
  # Set the TBBMALLOC_PLATFORM correctly so that find_package(TBBMalloc) sets the right hints
  set(TBBMALLOC_PLATFORM "mic")
elseif(HPX_PLATFORM_UC STREQUAL "BLUEGENEQ")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    hpx_error("HPX on the BG/Q can only be compiled with bgclang")
  endif()
  hpx_info("Compiling for BlueGene/Q")
endif()

################################################################################

################################################################################
# Add necessary compiler flags. Flags added here include flags to disable/enable
# certain warnings, enabling C++11 mode and disabling asserts. Setting of
# optimization flags is not handled here and is left to the responsibility of
# the user to avoid conflicts in the resulting binaries

hpx_add_compile_flag(-D_DEBUG
  CONFIGURATIONS Debug LANGUAGES CXX C Fortran)
hpx_add_compile_flag(-DDEBUG
  CONFIGURATIONS Debug LANGUAGES CXX C Fortran)
hpx_add_compile_flag(-DHPX_DISABLE_ASSERTS
  CONFIGURATIONS Release RelWithDebInfo MinSizeRelease LANGUAGES CXX C Fortran)
hpx_add_compile_flag(-DBOOST_DISABLE_ASSERTS
  CONFIGURATIONS Release RelWithDebInfo MinSizeRelease LANGUAGES CXX C Fortran)

# Make sure we compile in C++11 mode (MSVC uses it automatically)
if(NOT MSVC)
  #set(_cxx11_support FALSE)
  # Try -std=c++11 first
  check_cxx_compiler_flag(-std=c++11 WITH_CXX11)
  # ... If it failed try -std=c++0x
  if(NOT WITH_CXX11)
    check_cxx_compiler_flag(-std=c++0x WITH_CXX0X)
    if(WITH_CXX0X)
      set(_cxx11_flag -std=c++0x)
      set(WITH_CXX11 TRUE)
    endif()
  else()
    set(_cxx11_flag -std=c++11)
  endif()
  if(NOT WITH_CXX11)
    hpx_error("Unable to compile HPX without C++11 support in your compiler. HPX needs at the very least rvalue references with proper move semantics")
  endif()
  hpx_add_compile_flag(${_cxx11_flag})
endif()

if(MSVC)
  enable_language(ASM_MASM)
  hpx_add_compile_flag(-Ox
    CONFIGURATIONS Release LANGUAGES CXX C)
  hpx_add_compile_flag(-RTC1
    CONFIGURATIONS Debug LANGUAGES CXX C)

  # VS2012 and above has a special flag for improving the debug experience by
  # adding more symbol information to the build
  hpx_add_compile_flag(-d2ZI+
    CONFIGURATIONS RelWithDebInfo)

  # VS2013 and above know how to do link time constant data segment folding
  # VS2013 update 2 and above know how to remove debug information for
  # non-referenced functions and data (-Zc:inline)
  if(MSVC12)
    hpx_add_compile_flag(-Zc:inline)
    hpx_add_compile_flag(-Gw
      CONFIGURATIONS Release RelWithDebInfo MinSizeRelease)
  endif()
  # Exceptions
  hpx_add_compile_flag(-EHsc)
  # Runtime type information
  hpx_add_compile_flag(-GR)
  # Multiprocessor build
  hpx_add_compile_flag(-MP)
  # Increase the maximum size of object file sections
  hpx_add_compile_flag(-bigobj)
endif()

# Configure Warnings
if(WITH_WARNINGS)
  if(MSVC) # Adding special warning settings for the MSVC compiler ...
    hpx_add_compile_flag(-W3 LANGUAGES C CXX)
    # According to the ifort Windows manual, W3 isn't supported
    hpx_add_compile_flag(-W1 LANGUAGES Fortran)
    # Boost.Lockfree triggers 'warning C4307: '+' : integral constant overflow'
    # which is benign
    hpx_add_compile_flag(-wd4307)

    #MSVC2012/2013 are overeager to report 'qualifier applied to function type has no meaning; ignored'
    hpx_add_compile_flag(-wd4180)

    # OpenMPI triggers a lot of forcing value to bool 'true' or 'false' (performance warning)
    if(MPI_FOUND)
      hpx_add_compile_flag(-wd4800)
    endif()
  else() # Trial and error approach for any other compiler ...
    hpx_add_compile_flag_if_available(-Wall LANGUAGES CXX C Fortran)
    # This is a new warning popping up from the boost headers with no particular meaning
    hpx_add_compile_flag_if_available(-Wno-unused-local-typedefs LANGUAGES CXX C Fortran)
    hpx_add_compile_flag_if_available(-Wno-strict-aliasing LANGUAGES CXX C Fortran)
    hpx_add_compile_flag_if_available(-Wno-sign-promo LANGUAGES CXX)
    hpx_add_compile_flag_if_available(-Wno-attributes LANGUAGES CXX)
    hpx_add_compile_flag_if_available(-Wno-cast_align LANGUAGES CXX)
  endif()
endif()

# Diagnostics
if(MSVC)
  # Display full paths in diagnostics
  hpx_add_compile_flag(-FC LANGUAGES C CXX)
else()
  # Show the flags that toggle each warning
  hpx_add_compile_flag_if_available(-fdiagnostics-show-option LANGUAGES CXX C Fortran)

  # VLAs are a GNU extensions that we forbid as they are not supported on MSVC
  hpx_add_compile_flag_if_available(-Werror=vla)
  # No return statement in a non-void function can lead to garbage return values
  # in GCC.
  hpx_add_compile_flag_if_available(-Werror=return-type LANGUAGES CXX C)

  # We get false positives all over the place with this. Also, detection for
  # this flag fails with GCC 4.4 and 4.5.
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.5)
    hpx_add_compile_flag_if_available(-Wno-unused-but-set-parameter LANGUAGES CXX C)
    hpx_add_compile_flag_if_available(-Wno-unused-but-set-variable LANGUAGES CXX C)
    # Uninitialized variables are bad, earlier compilers issue spurious warnings
    hpx_add_compile_flag_if_available(-Werror=uninitialized LANGUAGES CXX C)
  endif()

  # Silence warning about __sync_fetch_and_nand changing semantics
  hpx_add_compile_flag_if_available(-Wno-sync-nand LANGUAGES CXX C)

  # Silence warnings about deleting polymorphic objects with non-virtual dtors.
  # These come from within Boost. Detection for this flag fails with GCC 4.4,
  # 4.5 and 4.6.
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.6)
    hpx_add_compile_flag_if_available(-Wno-delete-non-virtual-dtor LANGUAGES CXX C)
  endif()

  # Check if our libraries have unresolved symbols
  hpx_add_link_flag_if_available(-Wl,-z,defs)

  if("${HPX_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
    hpx_use_flag_if_available(-Wno-deprecated-register CXX C)
  endif()

  if(WITH_HIDDEN_VISIBILITY)
    hpx_add_compile_flag_if_available(-fvisibility=hidden LANGUAGES CXX C Fortran)
    hpx_add_link_flag_if_available(-fvisibility=hidden LANGUAGES CXX C Fortran)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    hpx_add_compile_flag_if_available(-Wno-cast-align)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # Disable the following warnings:
    # #1170: invalid redeclaration of nested class
    # #858: type qualifier on return type is meaningless
    # #1098: the qualifier on this friend declaration is ignored
    # #488: template parameter not used in declaring the parameter type
    # #2203: cast discards qualifiers from target type (needed for mvapich2
    # mpi header)
    hpx_use_flag_if_available(-wd1170)
    hpx_use_flag_if_available(-wd858)
    hpx_use_flag_if_available(-wd1098)
    hpx_use_flag_if_available(-wd488)
    hpx_use_flag_if_available(-wd2203)
    hpx_use_flag_if_available(-wd2536)
  endif()
endif()

##############################################################################

##############################################################################
# Check the availability of certain C++11 language features

hpx_check_for_cxx11_rvalue_references()

if(NOT WITH_CXX11_RVALUE_REFERENCES)
  hpx_error("HPX needs support for C++11 rvalue references")
endif()

hpx_check_for_cxx11_variadic_templates(
  DEFINITIONS HPX_HAVE_CXX11_VARIADIC_TEMPLATES)

hpx_check_for_cxx11_lambdas(
  DEFINITIONS HPX_HAVE_CXX11_LAMBDAS)

hpx_check_for_cxx11_auto(
  DEFINITIONS HPX_HAVE_CXX11_LAMBDAS)

hpx_check_for_cxx11_decltype(
  DEFINITIONS HPX_HAVE_CXX11_DECLTYPE)

hpx_check_for_cxx11_alias_templates(
  DEFINITIONS HPX_HAVE_CXX11_ALIAS_TEMPLATES)

hpx_check_for_cxx11_defaulted_deleted_functions(
  DEFINITIONS HPX_HAVE_CXX11_DEFAULTED_DELETED_FUNCTIONS)

hpx_check_for_cxx11_std_unique_ptr(
  DEFINITIONS HPX_HAVE_CXX11_STD_UNIQUE_PTR
              BOOST_LOCKFREE_HAVE_CXX11_STD_UNIQUE_PTR)

if(NOT WITH_UTIL_TUPLE)
  hpx_check_for_cxx11_std_tuple(DEFINITIONS HPX_HAVE_CXX11_STD_TUPLE)
endif()

if(NOT WITH_UTIL_BIND)
  hpx_check_for_cxx11_std_bind(DEFINITIONS HPX_HAVE_CXX11_STD_BIND)
endif()

if(NOT WITH_UTIL_FUNCTION)
  hpx_check_for_cxx11_std_bind(DEFINITIONS HPX_HAVE_CXX11_STD_FUNCTION)
endif()

hpx_check_for_cxx11_std_initializer_list(
  DEFINITIONS HPX_HAVE_CXX11_STD_INITIALIZER_LIST)

##############################################################################

################################################################################
# Check Build Options based on the found dependencies. We also check for errors
# with incompatible options with the currently selected platform.
#

if(WITH_GENERIC_COROUTINE_CONTEXT)
  # Check if we can use generic coroutine contexts without any problems
  if(NOT Boost_CONTEXT_FOUND)
    hpx_error("The usage of Boost.Context was selected but Boost.Context was not found (Version 1.51 or higher is required).")
  endif()
  if(HPX_PLATFORM_UC STREQUAL "BLUEGENEQ")
    if(Boost_VERSION LESS 105600)
      hpx_error("On BlueGene/Q, Boost.Context can only be used with a Boost >=1.56")
    endif()
  endif()
  hpx_add_config_define(HPX_WITH_GENERIC_CONTEXT_COROUTINES)
endif()

################################################################################
# Emulation of SwapContext on Windows
################################################################################
if(MSVC)
  if(WITH_SWAP_CONTEXT_EMULATION)
    if(NOT CMAKE_ASM_MASM_COMPILER)
      hpx_error("SwitchToFiber emulation can not be enabled. The masm compiler could not be found. Try setting the ASM_MASM environment variable to the assembler executable (ml.exe/ml64.exe) or disable the emulation by setting WITH_SWAP_CONTEXT_EMULATION to Off")
    else()
      hpx_info("SwitchToFiber emulation is enabled, using compiler: '${CMAKE_ASM_MASM_COMPILER}'")
    endif()
endif()

################################################################################


# ################################################################################
# # search path configuration
# ################################################################################
#
# if (NOT DEFINED LIB)
#   set(LIB "lib")
# endif(NOT DEFINED LIB)
# ###############################################################################
# if(UNIX)
#   hpx_link_directories("${CMAKE_BINARY_DIR}/lib/hpx")
# else()
#   hpx_link_directories("${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib/hpx")
# endif()

################################################################################
# Git commit detection
################################################################################
include(HPX_GitCommit)
hpx_add_config_define(HPX_GIT_COMMIT "\"${HPX_GIT_COMMIT}\"")
#
# ################################################################################
# # HPX_PREFIX
# ################################################################################
# # If you want to run hpx applications without installing them, set this to
# # ${CMAKE_BINARY_DIR} when using Makefiles or
# # ${CMAKE_BINARY_DIR}/$(Configuration) when using Visual Studio
# if("${HPX_PLATFORM}" STREQUAL "Android")
#   set(HPX_PREFIX "lib" CACHE PATH "Where the hpx applications look for component dlls" FORCE)
# else()
#   if(MSVC)
#       if(HPX_NO_INSTALL)
#         set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}" CACHE PATH "Prefix prepended to install directories." FORCE)
#       endif()
#       set(hpx_build_prefix "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
#   else()
#       set(hpx_build_prefix "${CMAKE_BINARY_DIR}")
#   endif()
#   if(NOT ("${hpx_build_prefix}" STREQUAL "${CMAKE_INSTALL_PREFIX}"))
#     set(hpx_build_prefix "${hpx_build_prefix};${CMAKE_INSTALL_PREFIX}")
#   endif()
#   set(HPX_PREFIX "${hpx_build_prefix}" CACHE PATH "Where the HPX applications look for component dlls" FORCE)
# endif()
#
# hpx_add_config_define(HPX_PREFIX "\"${HPX_PREFIX}\"")
#
# hpx_include(SetOutputPaths)
# ##############################################################################
#
################################################################################
# # Configure compression and other plugins
# ###############################################################################
# add_hpx_pseudo_target(plugins)
# add_subdirectory(plugins)
#
# if(HPX_HAVE_COMPRESSION_BZIP2 AND BZIP2_FOUND)
#   hpx_add_config_define(HPX_HAVE_COMPRESSION_BZIP2)
# endif()
# if(HPX_HAVE_COMPRESSION_SNAPPY AND SNAPPY_FOUND)
#   hpx_add_config_define(HPX_HAVE_COMPRESSION_SNAPPY)
# endif()
# if(HPX_HAVE_COMPRESSION_ZLIB AND ZLIB_FOUND)
#   hpx_add_config_define(HPX_HAVE_COMPRESSION_ZLIB)
# endif()
#
# # Parcel coalescing is used by the main HPX library, enable it always
# hpx_add_config_define(HPX_HAVE_PARCEL_COALESCING)
#
# ################################################################################
# # Documentation toolchain (DocBook, BoostBook, QuickBook, xsltproc)
# ################################################################################
# hpx_include(Documentation)
#
# ################################################################################
# # Creating the partially preprocessed header requires a recent version of Wave
# ################################################################################
# hpx_include(Preprocessing)
#
# ################################################################################
# # Compiler configuration
# ################################################################################
#
# # Debug library postfix
# set(CMAKE_DEBUG_POSTFIX "d")
# set(HPX_DEBUG_POSTFIX "d")
#
# ################################################################################
# # Utility configuration
# ################################################################################
#
# if(HPX_UTIL_BIND)
#   hpx_add_config_define(HPX_UTIL_BIND)
# endif()
#
# if(HPX_UTIL_FUNCTION)
#   hpx_add_config_define(HPX_UTIL_FUNCTION)
# endif()
#
# if(HPX_UTIL_TUPLE)
#   hpx_add_config_define(HPX_UTIL_TUPLE)
# endif()
#
# ################################################################################
# # Scheduler configuration
# ################################################################################
# if(HPX_LOCAL_SCHEDULER OR HPX_ALL_SCHEDULERS)
#   hpx_add_config_define(HPX_LOCAL_SCHEDULER)
# endif()
#
# if(HPX_STATIC_PRIORITY_SCHEDULER OR HPX_ALL_SCHEDULERS)
#   hpx_add_config_define(HPX_STATIC_PRIORITY_SCHEDULER)
# endif()
#
# if(HPX_HIERARCHY_SCHEDULER OR HPX_ALL_SCHEDULERS)
#   hpx_add_config_define(HPX_HIERARCHY_SCHEDULER)
# endif()
#
# if(HPX_PERIODIC_PRIORITY_SCHEDULER OR HPX_ALL_SCHEDULERS)
#   hpx_add_config_define(HPX_PERIODIC_PRIORITY_SCHEDULER)
# endif()
#
# if(HPX_GOOGLE_PERFTOOLS)
#   find_package(HPX_GooglePerftools)
#   if(GOOGLE_PERFTOOLS_FOUND)
#     set(hpx_LIBRARIES ${hpx_LIBRARIES} ${GOOGLE_PERFTOOLS_LIBRARY})
#     hpx_include_sys_directories("${GOOGLE_PERFTOOLS_INCLUDE_DIR}")
#     hpx_link_sys_directories("${GOOGLE_PERFTOOLS_LIBRARY_DIR}")
#     hpx_add_config_define(HPX_GOOGLE_PERFTOOLS)
#   endif()
# endif()
#
# ################################################################################
# # MSVC configuration
# ################################################################################
# if(MSVC)
#   set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} psapi shlwapi)
#
#   ##############################################################################
#   # Stacktraces
#   ##############################################################################
#   if(HPX_HAVE_STACKTRACES)
#     set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} dbghelp)
#   endif()
#
#   ##############################################################################
#   # C++11
#   ##############################################################################
#
#   ##############################################################################
#   # Macro definitions for system headers
#   ##############################################################################
#   hpx_add_definitions(-D_WINDOWS)
#   hpx_add_definitions(-D_WIN32)
#   hpx_add_config_define(_WIN32_WINNT 0x0601)
#   hpx_add_config_define(_SCL_SECURE_NO_WARNINGS)
#   hpx_add_config_define(_CRT_SECURE_NO_WARNINGS)
#   hpx_add_config_define(_SCL_SECURE_NO_DEPRECATE)
#   hpx_add_config_define(_CRT_SECURE_NO_DEPRECATE)
#   hpx_add_config_define(_CRT_NONSTDC_NO_WARNINGS)
#
#   ##############################################################################
#   # Boost
#   ##############################################################################
#   # We auto-link Boost on Windows, so don't specify boost libraries explicitly
#   set(BOOST_FOUND_LIBRARIES "")
#   set(BOOST_serialization_LIBRARY "")
#
#   hpx_add_config_define(BOOST_USE_WINDOWS_H)
#   hpx_add_config_define(BOOST_SERIALIZATION_DYN_LINK)
#   if(NOT HPX_USE_GENERIC_COROUTINE_CONTEXT)
#     hpx_add_config_define(HPX_HAVE_FIBER_BASED_COROUTINES)
#   endif()
#   hpx_add_config_define(PSAPI_VERSION 1)
#
# ################################################################################
# # GCC-compatible compiler configuration
# ################################################################################
#   if(APPLE)
#     hpx_option(HPX_STDLIB_IS_LIBCPP BOOL "Adds -stdlib=libc++ to the linker command by default" OFF ADVANCED)
#     if(HPX_STDLIB_IS_LIBCPP)
#       hpx_use_flag_if_available(-stdlib=libc++ LANGUAGES CXX)
#     endif()
#     hpx_use_flag_if_available(-ftemplate-depth=256 LANGUAGES CXX)
#   endif()

#   # cmpxchg16b is an x86-64 extension present on most newer x86-64 machines.
#   # It allows us to do double quadword (128bit) atomic compare and swap
#   # operations, which is AWESOME. Note that early x86-64 processors do lack
#   # this instruction.
#   hpx_use_flag_if_available(-mcx16 LANGUAGES CXX C Fortran)
#
#   if(HPX_CXX_FLAG_MCX16)
#     hpx_add_config_define(HPX_HAVE_GNU_SYNC_16)
#     hpx_add_config_define(BOOST_ATOMIC_HAVE_GNU_SYNC_16) # for the gnu code
#   endif()
#
#   # __attribute__ ((aligned(16))) should align a variable to a 16-byte, however,
#   # GCC sets an upper limit on the maximum alignment (__BIGGEST_ALIGNMENT__)
#   # and some versions don't warn if you ask for an alignment above said limit.
#   # Instead, they'll just silently use the maximum, which can be problematic.
#   hpx_check_for_gnu_aligned_16(HPX_GNU_ALIGNED_16
#     DEFINITIONS HPX_HAVE_GNU_ALIGNED_16
#                 BOOST_ATOMIC_HAVE_GNU_ALIGNED_16) # for the gnu code
#
#   # __uint128_t and __int128_t are a nifty, albeit undocumented, GNU extension
#   # that's been supported in GCC (4.1ish and up) and clang-linux for a while
#   # (strangely, Intel-linux doesn't support this). This is particularly useful
#   # for use with cmpxchg16b
#   hpx_check_for_gnu_128bit_integers(HPX_GNU_128BIT_INTEGERS
#     DEFINITIONS HPX_HAVE_GNU_128BIT_INTEGERS
#                 BOOST_ATOMIC_HAVE_GNU_128BIT_INTEGERS) # for integral casts
#
#   # rdtsc is an x86 instruction that reads the value of a CPU time stamp
#   # counter. rdtscp is an extension to rdtsc. The difference is that rdtscp is
#   # a serializing instruction.
#   hpx_cpuid("rdtsc" HPX_RDTSC
#     DEFINITIONS HPX_HAVE_RDTSC)
#   hpx_cpuid("rdtscp" HPX_RDTSCP
#     DEFINITIONS HPX_HAVE_RDTSCP)
#
#   if(NOT HPX_RDTSC AND NOT HPX_RDTSCP)
#     hpx_warn("hardware_timer" "Neither rdtsc nor rdtscp is available; some performance counters may report incorrect results")
#   endif()
#
#   hpx_option(HPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION BOOL
#       "Disable automatic registration with Boost.Serialization (Default: Off)" Off ADVANCED)
#
#   ##############################################################################
#   # Macro definitions for system headers
#   ##############################################################################
#   hpx_add_definitions(-D_GNU_SOURCE)
#
#   ##############################################################################
#   # System libraries
#   ##############################################################################
#   if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
#     set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} dl)
#   endif()
#
#   if(NOT APPLE AND NOT ("${HPX_PLATFORM}" STREQUAL "Android"))
#     set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} rt)
#   endif()
#
#   if("${HPX_PLATFORM}" STREQUAL "Android")
#     set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} log)
#   endif()
#
# endif()
#
# if(HPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION)
#   hpx_add_config_define(HPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION)
# endif()
#
# ###############################################################################
# # Deprecated future
# ###############################################################################
# if(HPX_HAVE_CXX11_ALIAS_TEMPLATES)
#   hpx_option(HPX_UNIQUE_FUTURE_ALIAS BOOL
#     "HPX will define unique_future<R> as a template alias of future<R> (default: OFF)."
#     OFF ADVANCED)
#
#   if(HPX_UNIQUE_FUTURE_ALIAS)
#     hpx_add_config_define(HPX_UNIQUE_FUTURE_ALIAS)
#   endif()
# endif()
#
# ################################################################################
# # Target specification
# ################################################################################
# # Recurse into some subdirectories. This does not actually cause another cmake
# # executable to run. The same process will walk through the project's entire
# # directory structure.
# add_subdirectory(src)
#
# hpx_option(HPX_BUILD_EXAMPLES BOOL "Build HPX examples (default: ON)" ON ADVANCED)
#
# if(HPX_BUILD_EXAMPLES)
#   add_hpx_pseudo_target(examples)
#   include_directories(examples)
#   add_subdirectory(examples)
# endif()
#
# ###############################################################################
# # Activate plugins
# ###############################################################################
# add_plugin_modules()
#
# ###############################################################################
#
# find_package(PythonInterp)
#
# if(NOT PYTHONINTERP_FOUND)
#     hpx_warn("hpx.tests" "Tests have been disabled because python was not found")
#     hpx_option(HPX_BUILD_TESTS BOOL "Build HPX tests (default: OFF)" OFF ADVANCED)
# else()
#     hpx_option(HPX_BUILD_TESTS BOOL "Build HPX tests (default: ON)" ON ADVANCED)
# endif()
#
# if(HPX_BUILD_TESTS)
#   hpx_option(HPX_BUILD_TESTS_BENCHMARKS BOOL "Build HPX benchmark tests (default: ON)" ON ADVANCED)
#   hpx_option(HPX_BUILD_TESTS_REGRESSIONS BOOL "Build HPX regression tests (default: ON)" ON ADVANCED)
#   hpx_option(HPX_BUILD_TESTS_UNIT BOOL "Build HPX unit tests (default: ON)" ON ADVANCED)
#
#   add_hpx_pseudo_target(tests)
#
#   enable_testing()
#   include(CTest)
#
#   add_custom_command(TARGET tests POST_BUILD
#     COMMAND ctest --output-on-failure --timeout 100)
#
#   include_directories(tests)
#   add_subdirectory(tests)
# else()
#   unset(HPX_BUILD_TESTS_BENCHMARKS CACHE)
#   unset(HPX_BUILD_TESTS_REGRESSIONS CACHE)
#   unset(HPX_BUILD_TESTS_UNIT CACHE)
# endif()
#
# hpx_option(HPX_BUILD_RUNTIME BOOL "Build HPX runtime (default: ON)" ON ADVANCED)
#
# if(HPX_BUILD_RUNTIME)
#   add_hpx_pseudo_target(runtime)
#   add_subdirectory(runtime)
# endif()
#
# hpx_option(HPX_BUILD_TOOLS BOOL "Build HPX tools (default: OFF)" OFF ADVANCED)
#
# if(HPX_BUILD_TOOLS)
#   add_hpx_pseudo_target(tools)
#   add_subdirectory(tools)
# endif()
#
# hpx_option(HPX_BUILD_DOCUMENTATION BOOL "The HPX documentation toolchain is available (default OFF)." OFF)
# if(HPX_BUILD_DOCUMENTATION)
#   hpx_option(HPX_BUILD_DOCUMENTATION_SINGLEPAGE BOOL "The HPX documentation should be build as a single page HTML (default OFF)." OFF)
#   add_subdirectory(docs)
# endif()
#
# if(${BOOST_MINOR_VERSION} GREATER 50)
#   add_subdirectory(preprocess)
# endif()
#
# ################################################################################
# # Configure the header to include all compile definitions
# ################################################################################
# list(SORT HPX_CONFIG_DEFINITIONS)
# list(REMOVE_DUPLICATES HPX_CONFIG_DEFINITIONS)
# set(hpx_config_defines "\n")
# foreach(def ${HPX_CONFIG_DEFINITIONS})
#     if("${def}" STREQUAL "HPX_PREFIX" AND NOT MSVC)
#       string(REPLACE ";" ":" define "${${def}_define}")
#       set(hpx_config_defines "${hpx_config_defines}#define ${def} ${define}\n")#"
#     else()
#       set(hpx_config_defines "${hpx_config_defines}#define ${def} ${${def}_define}\n")#"
#     endif()
# endforeach()
#
# configure_file("${hpx_SOURCE_DIR}/cmake/templates/config_defines.hpp.in"
#                "${CMAKE_BINARY_DIR}/hpx/config/defines.hpp"
#                @ONLY)
#
# # Configure hpxrun.py
# configure_file("${hpx_SOURCE_DIR}/cmake/templates/hpxrun.py.in"
#                "${CMAKE_BINARY_DIR}/bin/hpxrun.py"
#                @ONLY)
#
# ################################################################################
# # installation instructions
# ################################################################################
# if(NOT HPX_NO_INSTALL)
#   install( # install all hpx header files
#     DIRECTORY hpx/
#     DESTINATION include/hpx
#     COMPONENT core
#     FILES_MATCHING PATTERN "*.hpp"
#     PATTERN ".git" EXCLUDE
#     PATTERN "CMakeFiles" EXCLUDE
#     PATTERN "CTestFiles" EXCLUDE)
#
#   # Install all HPX header that have been configured using various
#   # cmake options
#   install(
#     DIRECTORY "${CMAKE_BINARY_DIR}/hpx/"
#     DESTINATION include/hpx
#     COMPONENT core
#     FILES_MATCHING PATTERN "*.hpp"
#     PATTERN ".git" EXCLUDE
#     PATTERN "CMakeFiles" EXCLUDE
#     PATTERN "CTestFiles" EXCLUDE)
#
#   install( # install all hpx header files
#     DIRECTORY hpx
#     DESTINATION share/hpx-${HPX_VERSION}/docs/html/code
#     COMPONENT core
#     FILES_MATCHING PATTERN "*.hpp"
#     PATTERN ".git" EXCLUDE
#     PATTERN "CMakeFiles" EXCLUDE
#     PATTERN "CTestFiles" EXCLUDE)
#
#   install( # install all hpx source files
#     DIRECTORY src
#     DESTINATION share/hpx-${HPX_VERSION}/docs/html/code
#     COMPONENT core
#     PATTERN ".git" EXCLUDE
#     PATTERN "CMakeFiles" EXCLUDE
#     PATTERN "CTestFiles" EXCLUDE)
#
#   install( # install all hpx examples source files
#     DIRECTORY examples
#     DESTINATION share/hpx-${HPX_VERSION}/docs/html/code
#     COMPONENT core
#     PATTERN ".git" EXCLUDE
#     PATTERN "CMakeFiles" EXCLUDE
#     PATTERN "CTestFiles" EXCLUDE)
#
#   install( # install all hpx tests source files
#     DIRECTORY tests
#     DESTINATION share/hpx-${HPX_VERSION}/docs/html/code
#     COMPONENT core
#     PATTERN ".git" EXCLUDE
#     PATTERN "CMakeFiles" EXCLUDE
#     PATTERN "CTestFiles" EXCLUDE)
#
#   install( # Install all HPX cmake utility files
#     DIRECTORY cmake/
#     DESTINATION share/hpx-${HPX_VERSION}/cmake
#     COMPONENT core
#     PATTERN ".cmake.in" EXCLUDE
#     PATTERN ".git" EXCLUDE)
#
#   install( # Install HPX Python scripts
#     DIRECTORY python/scripts/
#     DESTINATION bin
#     FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
#                      GROUP_READ GROUP_EXECUTE
#                      WORLD_READ WORLD_EXECUTE
#     COMPONENT core
#     FILES_MATCHING PATTERN "*.py"
#     PATTERN ".git" EXCLUDE)
#
#   if(UNIX)
#     file(GLOB scripts "${CMAKE_SOURCE_DIR}/python/scripts/*.py")
#     execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory
#       "${CMAKE_BINARY_DIR}/python/scripts" )
#     foreach(script ${scripts})
#       get_filename_component(script_name ${script} NAME)
#       get_filename_component(script_name_we ${script} NAME_WE)
#
#       #make copy, so that we have intact symlink in CMAKE_BINARY_DIR
#       execute_process(
#         COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${script}" "${script_name}"
# 	WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python/scripts")
#
#       execute_process(
#         COMMAND "${CMAKE_COMMAND}" -E create_symlink "${script_name}" "${script_name_we}"
# 	WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python/scripts")
#
#       install(FILES "${CMAKE_BINARY_DIR}/python/scripts/${script_name_we}" DESTINATION "bin")
#     endforeach()
#   endif()
#
#   install( # Install HPX Python module (TODO: this is a temporary hack)
#     DIRECTORY python/hpx
#     DESTINATION share/hpx-${HPX_VERSION}/python
#     COMPONENT core
#     FILES_MATCHING PATTERN "*.py"
#     PATTERN ".git" EXCLUDE)
#
#   execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/python" )
#   #make symlink, so that we have intact hpx symlink in CMAKE_BINARY_DIR
#   execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink "${CMAKE_SOURCE_DIR}/python/hpx"
#     "hpx-${HPX_VERSION}" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python")
#   # create a symlink in share pointing to the latest HPX installation
#   execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink "hpx-${HPX_VERSION}" "hpx" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python")
#   install(DIRECTORY "${CMAKE_BINARY_DIR}/python/hpx" DESTINATION "share")
#
#   install( # Install external dependencies
#     DIRECTORY external/cache/boost
#               external/endian/boost
#     DESTINATION include/hpx/external
#     COMPONENT core
#     FILES_MATCHING PATTERN "*.hpp"
#     PATTERN ".git" EXCLUDE)
#
#   if(${BOOST_MINOR_VERSION} LESS 49)
#     # Install Boost.Serialization patch for older Boost versions only
#     install(
#       DIRECTORY external/serialization/boost
#       DESTINATION include/hpx/external
#       COMPONENT core
#       FILES_MATCHING PATTERN "*.hpp"
#       PATTERN ".git" EXCLUDE)
#   endif()
#
#   if(HPX_NATIVE_MIC)
#     # FIXME: push changes upstream
#     install(
#       DIRECTORY external/asio/boost
#       DESTINATION include/hpx/external
#       COMPONENT core
#       FILES_MATCHING PATTERN "*.hpp"
#       PATTERN ".git" EXCLUDE)
#   endif()
#
#   if(${BOOST_MINOR_VERSION} LESS 53)
#     # Install Boost.Atomic library for older Boost versions only
#     install(
#       DIRECTORY external/atomic/boost
#       DESTINATION include/hpx/external
#       COMPONENT core
#       FILES_MATCHING PATTERN "*.hpp"
#       PATTERN ".git" EXCLUDE)
#   endif()
#
#   if(${BOOST_MINOR_VERSION} LESS 53)
#     # Install Boost.Lockfree library for older Boost versions only
#     install(
#       DIRECTORY external/lockfree/boost
#       DESTINATION include/hpx/external
#       COMPONENT core
#       FILES_MATCHING PATTERN "*.hpp"
#       PATTERN ".git" EXCLUDE)
#   endif()
#
#   install(
#     FILES "${hpx_SOURCE_DIR}/LICENSE_1_0.txt"
#     DESTINATION share/hpx-${HPX_VERSION}
#     COMPONENT license)
#
#   if(HPX_BUILD_DOCUMENTATION)
#     # Install HPX documentation files
#     if(MSVC)
#       set(doc_dir "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
#     else()
#       set(doc_dir "${CMAKE_BINARY_DIR}")
#     endif()
#
#     install(
#       DIRECTORY "${doc_dir}/share/hpx-${HPX_VERSION}/docs/"
#       DESTINATION share/hpx-${HPX_VERSION}/docs
#       COMPONENT docs
#       OPTIONAL
#       FILES_MATCHING PATTERN "*.html"
#       PATTERN "*.pdf"
#       PATTERN ".git" EXCLUDE)
#
#     install(
#       FILES "${hpx_SOURCE_DIR}/docs/index.html"
#       DESTINATION share/hpx-${HPX_VERSION}/
#       COMPONENT docs)
#
#     install(
#       DIRECTORY "${hpx_SOURCE_DIR}/docs/html/src/"
#       DESTINATION share/hpx-${HPX_VERSION}/docs/html/src
#       COMPONENT docs
#       FILES_MATCHING PATTERN "*.css"
#       PATTERN ".git" EXCLUDE)
#
#     install(
#       DIRECTORY "${hpx_SOURCE_DIR}/docs/html/images/"
#       DESTINATION share/hpx-${HPX_VERSION}/docs/html/images
#       COMPONENT docs
#       FILES_MATCHING PATTERN "*.png"
#       PATTERN ".git" EXCLUDE)
#   endif()
# endif()
#
# ################################################################################
# # External build system support (FindHPX.cmake and pkg-config).
# ################################################################################
# set(HPX_LINK_DIRECTORIES
#     "${CMAKE_INSTALL_PREFIX}/lib/hpx" ${HPX_LINK_DIRECTORIES})
# set(HPX_INCLUDE_DIRECTORIES
#     "${CMAKE_INSTALL_PREFIX}/include" ${HPX_INCLUDE_DIRECTORIES}
#     "${CMAKE_INSTALL_PREFIX}/include/hpx/external")
#
# list(REMOVE_DUPLICATES HPX_LINK_DIRECTORIES)
# list(REMOVE_DUPLICATES HPX_INCLUDE_DIRECTORIES)
#
# set(external_rpath "")
# set(external_link_directories "")
# set(external_libraries "")
# set(external_libraries_name "")
# set(external_link_flags "")
# set(external_include_directories "")
# set(external_include_flags "")
#
# foreach(directory ${HPX_LINK_DIRECTORIES})
#   string(REGEX REPLACE "([ ()])" "\\\\\\1" directory "${directory}")
#   set(external_link_directories "${external_link_directories} ${directory}")
#   set(external_link_flags "${external_link_flags} -L${directory}")
#   set(external_rpath "${external_rpath}:${directory}")
# endforeach()
#
# #hpx_info("external_link_directories" "hpx_LIBRARIES: ${hpx_LIBRARIES}")
# foreach(lib ${hpx_LIBRARIES})
#   get_filename_component(lib_dir ${lib} PATH)
#   if(lib_dir)
#     string(REGEX REPLACE "([ ()])" "\\\\\\1" lib_dir "${lib_dir}")
#     set(external_link_directories "${external_link_directories} ${lib_dir}")
#   endif()
#   get_filename_component(lib ${lib} NAME_WE)
#
#   # if a library starts with 'lib' on Windows, this is probably by design
#   if(NOT MSVC)
#     string(REGEX REPLACE "^lib" "" lib ${lib})
#   endif()
#
#   set(external_libraries "${external_libraries} ${lib}")
#   set(external_libraries_name "${external_libraries_name} -l${lib}")
# endforeach()
#
# # Make lists unique
# list(REMOVE_DUPLICATES external_link_directories)
# list(REMOVE_DUPLICATES external_libraries)
# list(REMOVE_DUPLICATES external_libraries_name)
#
# if(HPX_NO_INSTALL)
# #  foreach(PREFIX ${HPX_PREFIX})
# #    list(REMOVE_ITEM HPX_INCLUDE_DIRECTORIES "${PREFIX}/include)"
# #                                             "${PREFIX}/include/hpx/external")
# #  endforeach()
# else()
#   list(REMOVE_ITEM HPX_INCLUDE_DIRECTORIES "${hpx_SOURCE_DIR}"
#                                            "${CMAKE_BINARY_DIR}")
# endif()
#
# foreach(directory ${HPX_INCLUDE_DIRECTORIES})
#   string(REGEX REPLACE "([ ()])" "\\\\\\1" directory "${directory}")
#   set(external_include_directories "${external_include_directories} ${directory}")
#   set(external_include_flags "${external_include_flags} -I${directory}")
# endforeach()
#
# # Make lists unique
# list(REMOVE_DUPLICATES external_include_directories)
# list(REMOVE_DUPLICATES external_include_flags)
#
# set(boost_library_info "")
#
# foreach(library ${BOOST_LIBRARIES})
#   string(TOUPPER ${library} library_uc)
#   string(REGEX REPLACE "([ ()])" "\\\\\\1" "BOOST_${library_uc}_LIBRARY" "${BOOST_${library_uc}_LIBRARY}")
#   set(boost_library_info
#     "${boost_library_info}set(HPX_BOOST_${library_uc}_LIBRARY ${BOOST_${library_uc}_LIBRARY})\n")
# # FIXME: this doesn't work when used in external projects
# #  set(boost_library_info
# #    "${boost_library_info}    CACHE FILEPATH \"Boost ${library} shared library.\")\n")
# endforeach()
#
# set(external_definitions "")
#
# foreach(definition ${HPX_DEFINITIONS})
#     set(external_definitions "${external_definitions} ${definition}")
# endforeach()
#
# # Make list unique
# list(REMOVE_DUPLICATES external_definitions)
#
# set(CMAKE_DIR "cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" CACHE STRING "directory (in share), where to put FindHPX cmake module")
#
# if(MSVC)
#   set(output_dir "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
# else()
#   set(output_dir "${CMAKE_BINARY_DIR}")
# endif()
#
# string(REGEX REPLACE "([ ()])" "\\\\\\1" HPX_PREFIX "${HPX_PREFIX}")
# if(HPX_NO_INSTALL)
#   set(external_cmake_path "${hpx_SOURCE_DIR}/cmake")
# else()
#     set(external_cmake_path "${CMAKE_INSTALL_PREFIX}/share/hpx-${HPX_VERSION}/cmake")
# endif()
#
# set(hpx_link_flag_target_properties "")
# if(MSVC AND HPX_LINK_FLAG_TARGET_PROPERTIES)
#   set(hpx_link_flag_target_properties "${HPX_LINK_FLAG_TARGET_PROPERTIES}")
# endif()
#
# configure_file("${hpx_SOURCE_DIR}/cmake/templates/FindHPX.cmake.in"
#                "${output_dir}/share/${CMAKE_DIR}/Modules/FindHPX.cmake"
#                ESCAPE_QUOTES @ONLY)
#
# if(NOT HPX_NO_INSTALL)
#   install(FILES "${output_dir}/share/${CMAKE_DIR}/Modules/FindHPX.cmake"
#           DESTINATION "share/${CMAKE_DIR}/Modules")
# endif()
#
# if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
#     configure_file("${hpx_SOURCE_DIR}/cmake/templates/hpx_application.pc.in"
#                    "${output_dir}/lib/pkgconfig/hpx_application.pc"
#                    @ONLY)
#     configure_file("${hpx_SOURCE_DIR}/cmake/templates/hpx_component.pc.in"
#                    "${output_dir}/lib/pkgconfig/hpx_component.pc"
#                    @ONLY)
# else()
#     configure_file("${hpx_SOURCE_DIR}/cmake/templates/hpx_application_debug.pc.in"
#                    "${output_dir}/lib/pkgconfig/hpx_application_debug.pc"
#                    @ONLY)
#
#     configure_file("${hpx_SOURCE_DIR}/cmake/templates/hpx_component_debug.pc.in"
#                    "${output_dir}/lib/pkgconfig/hpx_component_debug.pc"
#                    @ONLY)
# endif()
#
# if(NOT HPX_NO_INSTALL)
#   if(NOT ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug"))
#     install(FILES "${output_dir}/lib/pkgconfig/hpx_application.pc"
#             DESTINATION ${LIB}/pkgconfig)
#     install(FILES "${output_dir}/lib/pkgconfig/hpx_component.pc"
#             DESTINATION ${LIB}/pkgconfig)
#   else()
#     install(FILES "${output_dir}/lib/pkgconfig/hpx_application_debug.pc"
#             DESTINATION ${LIB}/pkgconfig)
#     install(FILES "${output_dir}/lib/pkgconfig/hpx_component_debug.pc"
#             DESTINATION ${LIB}/pkgconfig)
#   endif()
# endif()
#
# # Create a symlink in share pointing to the latest HPX installation
# execute_process(COMMAND
#   "${CMAKE_COMMAND}" -E create_symlink "hpx-${HPX_VERSION}" "hpx"
#   WORKING_DIRECTORY "${output_dir}/share")

set(_HPX_INFO_STRING "\nHPX has been configured with the following options:\n\n")

get_cmake_property(_variableNames CACHE_VARIABLES)
foreach(_cat "Generic" "Thread Manager" "AGAS" "Parcelport" "Profiling" "Debugging" "Security")

  set(_HPX_INFO_STRING "${_HPX_INFO_STRING}${_cat} Options:\n\n")

  foreach (_variableName ${_variableNames})
    get_property(_var_cat_strings CACHE "${_variableName}" PROPERTY STRINGS)
    set(_is_hpx_option FALSE)
    foreach(_v ${_var_cat_strings})
      if(_v STREQUAL "HPX_OPTION")
        set(_is_hpx_option TRUE)
      endif()
      if(_is_hpx_option)
        set(_var_cat ${_v})
      endif()
    endforeach()
    if(_is_hpx_option AND _var_cat)
      if(_cat STREQUAL _var_cat)
        get_property(_doc CACHE "${_variableName}" PROPERTY HELPSTRING)
        set(_HPX_INFO_STRING "${_HPX_INFO_STRING}  ${_variableName} = ${${_variableName}}\n")
        set(_HPX_INFO_STRING "${_HPX_INFO_STRING}     ${_doc}\n")
        set(_HPX_INFO_STRING "${_HPX_INFO_STRING}\n")
      endif()
    endif()
  endforeach()
endforeach()

# add_library(test_static STATIC t.cpp)
add_library(test_shared SHARED t.cpp)
add_executable(test t.cpp)
message(${HPX_LIBRARIES})
target_link_libraries(test test_shared ${HPX_LIBRARIES})

# message(${_HPX_INFO_STRING})
# message("HPX will be installed to ${CMAKE_INSTALL_PREFIX}")
