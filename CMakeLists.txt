# Copyright (c) 2007-2015 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
# Copyright (c)      2013 Jeroen Habraken
# Copyright (c)      2014 Andreas Schaefer
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least CMake V2.8.10
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

# explicitly set certain policies
cmake_policy(VERSION 2.8.12)
if(NOT(CMAKE_VERSION VERSION_LESS 3.1.0))
  cmake_policy(SET CMP0054 OLD)
  cmake_policy(SET CMP0042 NEW)
endif()

# Overrides must go before the project() statement, otherwise they are ignored.

################################################################################
# C++ overrides
################################################################################
set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HPX_CXXOverrides.cmake")

################################################################################
# Fortran overrides
################################################################################
set(CMAKE_USER_MAKE_RULES_OVERRIDE_Fortran
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HPX_FortranOverrides.cmake")

################################################################################
# Build type (needs to be handled before project command below)
################################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)")
endif()

################################################################################
# project metadata
################################################################################
project(HPX CXX C)

set(HPX_MAJOR_VERSION 0)
set(HPX_MINOR_VERSION 9)
set(HPX_PATCH_LEVEL   12)
set(HPX_VERSION "${HPX_MAJOR_VERSION}.${HPX_MINOR_VERSION}.${HPX_PATCH_LEVEL}")
set(HPX_LIBRARY_VERSION "${HPX_VERSION}")
set(HPX_SOVERSION ${HPX_MAJOR_VERSION})
set(HPX_PACKAGE_NAME HPX)

if(MSVC)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

################################################################################
# CMake configuration
################################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# We safe the passed compiler flag to a special variable. This is needed for our
# build system unit tests. Some flags might influence the created symbols (
# _GLIBCXX_DEBUG i look at you)
set(CMAKE_CXX_FLAGS_SAFE ${CMAKE_CXX_FLAGS})

include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

# include additional macro definitions
include(HPX_Utils)
include(HPX_ExportTargets)
include(HPX_Libraries)
include(HPX_LibraryDir)
include(HPX_AddConfigTest)
include(HPX_AddDefinitions)

hpx_force_out_of_tree_build("This project requires an out-of-source-tree build. See README.rst. Clean your CMake cache and CMakeFiles if this message persists.")

if(NOT HPX_CMAKE_LOGLEVEL)
  set(HPX_CMAKE_LOGLEVEL "WARN")
endif()

################################################################################
# Fortran compiler detection
#
hpx_option(HPX_WITH_FORTRAN
  BOOL
  "Enable or disable the compilation of Fortran examples using HPX"
  OFF ADVANCED)

if (HPX_WITH_FORTRAN)
  include(HPX_FortranCompiler)
endif()

################################################################################

################################################################################
# Setup platform for which HPX should be compiled for.
#
include(HPX_SetPlatform)
if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
  unset(HPX_LIBRARY_VERSION)
  unset(HPX_SOVERSION)
endif()
################################################################################

################################################################################
# Set our build options cache variables which are customizable by users
#

## Generic build options
set(DEFAULT_MALLOC "system")
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(DEFAULT_MALLOC "tcmalloc")
#    set(DEFAULT_MALLOC "jemalloc")
endif()

hpx_option(HPX_WITH_MALLOC
  STRING
  "Define which allocator should be linked in. Options are: system, tcmalloc, jemalloc, tbbmalloc, and custom (default is: tcmalloc)"
  ${DEFAULT_MALLOC} ADVANCED)

hpx_option(HPX_WITH_HWLOC
  BOOL
  "Use Hwloc for hardware topolgy information and thread pinning. If disabled, performance might be reduced."
  ON ADVANCED)

hpx_option(HPX_WITH_SECURITY BOOL "Enable security support via libsodium." OFF)

# Logging configuration
hpx_option(HPX_WITH_LOGGING BOOL
  "Build HPX with logging enabled (default: ON)."
  ON ADVANCED)
if(HPX_WITH_LOGGING)
  hpx_add_config_define(HPX_HAVE_LOGGING)
endif()

## Compiler related build options
hpx_option(HPX_WITH_GCC_VERSION_CHECK BOOL
  "Don't ignore version reported by gcc (default: ON)"
  ON ADVANCED)

hpx_option(HPX_WITH_DEFAULT_TARGETS BOOL
  "Associate the core HPX library with the default build target (default: ON)."
  ON ADVANCED CATEGORY "Build Targets")

hpx_option(HPX_WITH_COMPILER_WARNINGS BOOL
  "Enable compiler warnings (default: ON)"
  ON ADVANCED)

hpx_option(HPX_WITH_DOCUMENTATION BOOL
  "Build the HPX documentation (default OFF)."
  OFF CATEGORY "Build Targets")

hpx_option(HPX_WITH_DOCUMENTATION_SINGLEPAGE BOOL
  "The HPX documentation should be build as a single page HTML (default OFF)."
  OFF CATEGORY "Build Targets")

if(MSVC)
  hpx_option(HPX_WITH_PSEUDO_DEPENDENCIES BOOL
    "Force creating pseudo targets and pseudo dependencies (default OFF)."
    OFF CATEGORY "Build Targets")
else()
  hpx_option(HPX_WITH_PSEUDO_DEPENDENCIES BOOL
    "Force creating pseudo targets and pseudo dependencies (default ON)."
    ON CATEGORY "Build Targets")
endif()

################################################################################
# Some platforms do not support dynamic linking. Enable this to link all
# libraries statically. This also changes some of the internals of HPX related
# to how components are loaded.
################################################################################
hpx_option(HPX_WITH_STATIC_LINKING BOOL
  "Compile HPX statically linked libraries (Default: OFF)" OFF ADVANCED)
if(HPX_WITH_STATIC_LINKING)
  hpx_add_config_define(HPX_HAVE_STATIC_LINKING)
  set(hpx_library_link_mode STATIC)
  set(CMAKE_SHARED_LIBS OFF)
else()
  set(hpx_library_link_mode SHARED)
endif()

################################################################################

hpx_option(HPX_WITH_EXAMPLES BOOL "Build the HPX examples (default ON)" ON CATEGORY "Build Targets")
hpx_option(HPX_WITH_TESTS BOOL "Build the HPX tests (default ON)" ON CATEGORY "Build Targets")
hpx_option(HPX_WITH_TESTS_BENCHMARKS BOOL "Build HPX benchmark tests (default: ON)" ON ADVANCED CATEGORY "Build Targets")
hpx_option(HPX_WITH_TESTS_REGRESSIONS BOOL "Build HPX regression tests (default: ON)" ON ADVANCED CATEGORY "Build Targets")
hpx_option(HPX_WITH_TESTS_UNIT BOOL "Build HPX unit tests (default: ON)" ON ADVANCED CATEGORY "Build Targets")
hpx_option(HPX_WITH_TESTS_HEADERS BOOL "Build HPX header tests (default: OFF)" OFF ADVANCED CATEGORY "Build Targets")
hpx_option(HPX_WITH_TESTS_EXTERNAL_BUILD BOOL "Build external cmake build tests (default: ON)" ON ADVANCED CATEGORY "Build Targets")
hpx_option(HPX_WITH_TOOLS BOOL "Build HPX tools (default: OFF)" OFF ADVANCED CATEGORY "Build Targets")
hpx_option(HPX_WITH_RUNTIME BOOL "Build HPX runtime (default: ON)" ON ADVANCED CATEGORY "Build Targets")

hpx_option(HPX_WITH_COMPILE_ONLY_TESTS BOOL
  "Create build system support for compile time only HPX tests (default ON)"
  ON CATEGORY "Build Targets")

# Enable IO-counters on linux systems only
set(HPX_WITH_IO_COUNTERS_DEFAULT OFF)
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(HPX_WITH_IO_COUNTERS_DEFAULT ON)
endif()

hpx_option(HPX_WITH_IO_COUNTERS BOOL
  "Build HPX runtime (default: ${HPX_WITH_IO_COUNTERS_DEFAULT})"
  ${HPX_WITH_IO_COUNTERS_DEFAULT} ADVANCED CATEGORY "Build Targets")
if(HPX_WITH_IO_COUNTERS)
  hpx_add_config_define(HPX_HAVE_IO_COUNTERS)
endif()

set(HPX_FULL_RPATH_DEFAULT ON)
if(APPLE OR WIN32)
  set(HPX_FULL_RPATH_DEFAULT OFF)
endif()
hpx_option(HPX_WITH_FULL_RPATH BOOL
  "Build and link HPX libraries and executables with full RPATHs (default: ON)"
  OFF ADVANCED)

################################################################################
# HPX Compute configuration
################################################################################
hpx_option(HPX_WITH_CUDA BOOL
  "Enable CUDA support (default: OFF)" OFF ADVANCED)
hpx_option(HPX_WITH_CUDA_CLANG BOOL
  "Use clang to compile CUDA code (default: OFF)" OFF ADVANCED)
if(HPX_WITH_CUDA)
  find_package(CUDA REQUIRED)
  set(HPX_WITH_COMPUTE On)
  hpx_add_config_define(HPX_HAVE_CUDA)
  hpx_add_config_define(HPX_HAVE_COMPUTE)
endif()
hpx_option(HPX_WITH_HCC BOOL
  "Enable hcc support (default: OFF)" ADVANCED)
hpx_option(HPX_WITH_SYCL BOOL
  "Enable sycl support (default: OFF)" ADVANCED)

################################################################################
# Native TLS configuration
################################################################################
set(HPX_WITH_NATIVE_TLS_DEFAULT ON)
if(APPLE)
  set(HPX_WITH_NATIVE_TLS_DEFAULT OFF)
endif()
hpx_option(HPX_WITH_NATIVE_TLS BOOL
  "Use native TLS support if available (default: ${HPX_WITH_NATIVE_TLS_DEFAULT})"
  ${HPX_WITH_NATIVE_TLS_DEFAULT} ADVANCED)
if(HPX_WITH_NATIVE_TLS)
  hpx_info("Native TLS is enabled.")
  hpx_add_config_define(HPX_HAVE_NATIVE_TLS)
else()
  hpx_info("Native TLS is disabled.")
endif()

################################################################################
# Utility configuration
################################################################################
set(HPX_HIDDEN_VISIBILITY_DEFAULT ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.6.99
      AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
    set(HPX_HIDDEN_VISIBILITY_DEFAULT OFF)
  endif()
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

hpx_option(HPX_WITH_HIDDEN_VISIBILITY BOOL
  "Use -fvisibility=hidden for builds on platforms which support it (default ${HPX_HIDDEN_VISIBILITY_DEFAULT})"
  ${HPX_HIDDEN_VISIBILITY_DEFAULT}
  ADVANCED)

hpx_option(HPX_WITH_AUTOMATIC_SERIALIZATION_REGISTRATION BOOL
  "Use automatic serialization registration for actions and functions. This affects compatibility between HPX applications compiled with different compilers (default ON)"
  ON
  ADVANCED)
if(HPX_WITH_AUTOMATIC_SERIALIZATION_REGISTRATION)
  hpx_add_config_define(HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION)
endif()

## Thread Manager related build options

set(HPX_MAX_CPU_COUNT_DEFAULT "64")
hpx_option(HPX_WITH_MAX_CPU_COUNT STRING
  "HPX applications will not use more that this number of OS-Threads (default: ${HPX_MAX_CPU_COUNT_DEFAULT})"
  ${HPX_MAX_CPU_COUNT_DEFAULT}
  CATEGORY "Thread Manager" ADVANCED)
hpx_add_config_define(HPX_HAVE_MAX_CPU_COUNT ${HPX_WITH_MAX_CPU_COUNT})

hpx_option(HPX_WITH_MORE_THAN_64_THREADS BOOL
  "HPX applications will be able to run on more than 64 cores"
  (NOT (${HPX_WITH_MAX_CPU_COUNT} LESS 65))
  CATEGORY "Thread Manager" ADVANCED)
if(HPX_WITH_MORE_THAN_64_THREADS)
  hpx_add_config_define(HPX_HAVE_MORE_THAN_64_THREADS)
endif()

hpx_option(HPX_WITH_THREAD_STACK_MMAP BOOL
  "Use mmap for stack allocation on appropriate platforms"
  ON
  CATEGORY "Thread Manager" ADVANCED)

hpx_option(HPX_WITH_THREAD_MANAGER_IDLE_BACKOFF BOOL
  "HPX scheduler threads are backing off on idle queues (default: ON)"
  ON
  CATEGORY "Thread Manager" ADVANCED)

hpx_option(HPX_WITH_STACKTRACES BOOL
  "Attach backtraces to HPX exceptions (default: ON)"
  ON
  CATEGORY "Thread Manager" ADVANCED)

hpx_option(HPX_WITH_THREAD_BACKTRACE_ON_SUSPENSION BOOL
  "Enable thread stack back trace being captured on suspension (default: OFF)"
  OFF
  CATEGORY "Thread Manager" ADVANCED)

if(HPX_WITH_STACKTRACES OR HPX_WITH_THREAD_BACKTRACE_ON_SUSPENSION)
  hpx_info("Stack traces are enabled.")
  hpx_add_config_define(HPX_HAVE_STACKTRACES)
  if(WIN32)
    hpx_libraries(dbghelp)
  endif()

  hpx_option(HPX_WITH_THREAD_BACKTRACE_DEPTH STRING
    "Thread stack back trace depth being captured (default: 5)"
    "5"
    CATEGORY "Thread Manager" ADVANCED)
  hpx_add_config_define(HPX_HAVE_THREAD_BACKTRACE_DEPTH
    ${HPX_WITH_THREAD_BACKTRACE_DEPTH})
endif()

if(HPX_WITH_THREAD_BACKTRACE_ON_SUSPENSION)
  hpx_add_config_define(HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION)

  hpx_option(HPX_WITH_THREAD_FULLBACKTRACE_ON_SUSPENSION BOOL
    "Enable thread stack back trace being captured on suspension (default: OFF)"
    OFF
    CATEGORY "Thread Manager" ADVANCED)
  if(HPX_WITH_THREAD_FULLBACKTRACE_ON_SUSPENSION)
    hpx_add_config_define(HPX_HAVE_THREAD_FULLBAKCTRACE_ON_SUSPENSION)
  endif()
endif()

hpx_option(HPX_WITH_THREAD_TARGET_ADDRESS BOOL
  "Enable storing target address in thread for NUMA awareness (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

if(HPX_WITH_THREAD_TARGET_ADDRESS)
  hpx_add_config_define(HPX_HAVE_THREAD_TARGET_ADDRESS)
endif()

hpx_option(HPX_WITH_THREAD_QUEUE_WAITTIME BOOL
  "Enable collecting queue wait times for threads (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

if(HPX_WITH_THREAD_QUEUE_WAITTIME)
  hpx_add_config_define(HPX_HAVE_THREAD_QUEUE_WAITTIME)
endif()

hpx_option(HPX_WITH_THREAD_IDLE_RATES BOOL
  "Enable measuring the percentage of overhead times spent in the scheduler (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

hpx_option(HPX_WITH_THREAD_CREATION_AND_CLEANUP_RATES BOOL
  "Enable measuring thread creation and cleanup times (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

if(HPX_WITH_THREAD_IDLE_RATES)
  hpx_add_config_define(HPX_HAVE_THREAD_IDLE_RATES)
  if(HPX_WITH_THREAD_CREATION_AND_CLEANUP_RATES)
    hpx_add_config_define(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
  endif()
endif()

hpx_option(HPX_WITH_THREAD_CUMULATIVE_COUNTS BOOL
  "Enable keeping track of cumulative thread counts in the schedulers (default: ON)"
  ON CATEGORY "Thread Manager" ADVANCED)

if(HPX_WITH_THREAD_CUMULATIVE_COUNTS)
  hpx_add_config_define(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
endif()

hpx_option(HPX_WITH_THREAD_STEALING_COUNTS BOOL
  "Enable keeping track of counts of thread stealing incidents in the schedulers (default: ON)"
  ON CATEGORY "Thread Manager" ADVANCED)

if(HPX_WITH_THREAD_STEALING_COUNTS)
  hpx_add_config_define(HPX_HAVE_THREAD_STEALING_COUNTS)
endif()

hpx_option(HPX_WITH_THREAD_LOCAL_STORAGE BOOL
  "Enable thread local storage for all HPX threads (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

if(HPX_WITH_THREAD_LOCAL_STORAGE)
  hpx_add_config_define(HPX_HAVE_THREAD_LOCAL_STORAGE)
endif()

hpx_option(HPX_WITH_SCHEDULER_LOCAL_STORAGE BOOL
  "Enable scheduler local storage for all HPX schedulers (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

if(HPX_WITH_SCHEDULER_LOCAL_STORAGE)
  hpx_add_config_define(HPX_HAVE_SCHEDULER_LOCAL_STORAGE)
endif()

hpx_option(HPX_WITH_SWAP_CONTEXT_EMULATION BOOL
  "Emulate SwapContext API for coroutines (default: OFF)"
  OFF CATEGORY "Thread Manager" ADVANCED)

################################################################################
# Scheduler configuration
################################################################################
hpx_option(HPX_WITH_THREAD_SCHEDULERS STRING
  "Which thread schedulers are build. Options are: all, abp-priority, local, static-priority, static, hierarchy, and periodic-priority. For multiple enabled schedulers, separate with a semicolon (default: all)"
  "all"
  CATEGORY "Thread Manager" ADVANCED)

string(TOUPPER ${HPX_WITH_THREAD_SCHEDULERS} HPX_WITH_THREAD_SCHEDULERS_UC)
foreach(_scheduler ${HPX_WITH_THREAD_SCHEDULERS_UC})
  if(_scheduler STREQUAL "ALL")
    set(_all On)
    set(HPX_HAVE_ALL_SCHEDULERS ON CACHE INTERNAL "")
  endif()
  if(_scheduler STREQUAL "ABP-PRIORITY" OR _all)
    hpx_add_config_define(HPX_HAVE_ABP_SCHEDULER)
    set(HPX_HAVE_ABP_SCHEDULER ON CACHE INTERNAL "")
  endif()
  if(_scheduler STREQUAL "LOCAL" OR _all)
    hpx_add_config_define(HPX_HAVE_LOCAL_SCHEDULER)
    set(HPX_HAVE_LOCAL_SCHEDULER ON CACHE INTERNAL "")
  endif()
  if(_scheduler STREQUAL "STATIC-PRIORITY" OR _all)
    hpx_add_config_define(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    set(HPX_HAVE_STATIC_PRIORITY_SCHEDULER ON CACHE INTERNAL "")
  endif()
  if(_scheduler STREQUAL "STATIC" OR _all)
    hpx_add_config_define(HPX_HAVE_STATIC_SCHEDULER)
    set(HPX_HAVE_STATIC_SCHEDULER ON CACHE INTERNAL "")
  endif()
  # The throttle scheduler is not supported neither on Windows nor on Mac
  if(HPX_WITH_APEX AND NOT WIN32 AND NOT APPLE)
    if(_scheduler STREQUAL "THROTTLE" OR _all)
      hpx_add_config_define(HPX_HAVE_THROTTLE_SCHEDULER)
      set(HPX_HAVE_THROTTLE_SCHEDULER ON CACHE INTERNAL "")
    endif()
  endif()
  if(_scheduler STREQUAL "HIERARCHY" OR _all)
    hpx_add_config_define(HPX_HAVE_HIERARCHY_SCHEDULER)
    set(HPX_HAVE_HIERARCHY_SCHEDULER ON CACHE INTERNAL "")
  endif()
  if(_scheduler STREQUAL "PERIODIC-PRIORITY" OR _all)
    hpx_add_config_define(HPX_HAVE_PERIODIC_PRIORITY_SCHEDULER)
    set(HPX_HAVE_PERIODIC_PRIORITY_SCHEDULER ON CACHE INTERNAL "")
  endif()
  unset(_all)
endforeach()

## AGAS related build options
hpx_option(HPX_WITH_AGAS_DUMP_REFCNT_ENTRIES BOOL
  "Enable dumps of the AGAS refcnt tables to logs (default: OFF)"
  OFF CATEGORY "AGAS" ADVANCED)
if(HPX_WITH_AGAS_DUMP_REFCNT_ENTRIES)
  hpx_add_config_define(HPX_HAVE_AGAS_DUMP_REFCNT_ENTRIES)
endif()

## Parcelport related build options
hpx_option(HPX_WITH_PARCELPORT_IBVERBS BOOL
  "Enable the ibverbs based parcelport. This is currently an experimental feature"
  OFF CATEGORY "Parcelport" ADVANCED)
hpx_option(HPX_WITH_PARCELPORT_IPC BOOL
  "Enable the IPC (inter process communication) based parcelport. This is currently an experimental feature"
  OFF CATEGORY "Parcelport" ADVANCED)
hpx_option(HPX_WITH_PARCELPORT_MPI BOOL
  "Enable the MPI based parcelport."
  OFF CATEGORY "Parcelport")
hpx_option(HPX_WITH_PARCELPORT_TCP BOOL
  "Enable the TCP based parcelport."
  ON CATEGORY "Parcelport")

## ibverbs parcelport settings
hpx_option(HPX_WITH_PARCELPORT_IBVERBS_IFNAME STRING
  "The interface name of the ibverbs capable network adapter (default: ib0)"
  "ib0" CATEGORY "Parcelport" ADVANCED)
hpx_option(HPX_WITH_PARCELPORT_IBVERBS_MESSAGE_PAYLOAD STRING
  "Size of the message payload not sent with RDMA (default: 512 byte)"
  "512" CATEGORY "Parcelport" ADVANCED)
hpx_option(HPX_WITH_PARCELPORT_IBVERBS_MEMORY_CHUNK_SIZE STRING
  "Number of bytes a chunk in the memory pool can hold (default: 64MB)"
  "67108864" CATEGORY "Parcelport" ADVANCED)
hpx_option(HPX_WITH_PARCELPORT_IBVERBS_MAX_MEMORY_CHUNKS STRING
  "Maximum number of chunks that can be allocated (default: 100)"
  "100" CATEGORY "Parcelport" ADVANCED)

## mpi parcelport settings
hpx_option(HPX_WITH_PARCELPORT_MPI_ENV STRING
  "List of environment variables checked to detect MPI (default: MV2_COMM_WORLD_RANK;PMI_RANK;OMPI_COMM_WORLD_SIZE;ALPS_APP_PE)."
  "MV2_COMM_WORLD_RANK;PMI_RANK;OMPI_COMM_WORLD_SIZE;ALPS_APP_PE" CATEGORY "Parcelport" ADVANCED)
hpx_option(HPX_WITH_PARCELPORT_MPI_MULTITHREADED BOOL
  "Turn on MPI multithreading support (default: ON)."
  ON CATEGORY "Parcelport" ADVANCED)

## Profiling related build options
hpx_option(HPX_WITH_APEX BOOL "Enable APEX instrumentation support." OFF CATEGORY "Profiling")
if(HPX_WITH_APEX)
  hpx_add_config_define(HPX_HAVE_APEX)
endif()
hpx_option(HPX_WITH_PAPI BOOL "Enable the PAPI based performance counter." OFF CATEGORY "Profiling")
if(HPX_WITH_PAPI)
  hpx_add_config_define(HPX_HAVE_PAPI)
endif()
hpx_option(HPX_WITH_TAU BOOL "Enable TAU profiling support." OFF CATEGORY "Profiling")
if(HPX_WITH_TAU)
  hpx_add_config_define(HPX_HAVE_TAU)
endif()
hpx_option(HPX_WITH_GOOGLE_PERFTOOLS BOOL "Enable Google Perftools instrumentation support." OFF CATEGORY "Profiling")
if(HPX_WITH_GOOGLE_PERFTOOLS)
  hpx_add_config_define(HPX_HAVE_GOOGLE_PERFTOOLS)
endif()
hpx_option(HPX_WITH_ITTNOTIFY BOOL "Enable Amplifier (ITT) instrumentation support." OFF CATEGORY "Profiling")
#if(HPX_WITH_ITTNOTIFY)
  #hpx_add_config_define(HPX_HAVE_ITTNOTIFY)
#endif()

## Debugging related build options
hpx_option(HPX_WITH_VALGRIND BOOL "Enable Valgrind instrumentation support." OFF CATEGORY "Debugging")

hpx_option(HPX_WITH_VERIFY_LOCKS BOOL
  "Enable lock verification code (default: OFF, implicitly enabled in debug builds)"
  OFF
  CATEGORY "Debugging" ADVANCED)
hpx_option(HPX_WITH_VERIFY_LOCKS_GLOBALLY BOOL
  "Enable global lock verification code (default: OFF, implicitly enabled in debug builds)"
  OFF
  CATEGORY "Debugging" ADVANCED)
hpx_option(HPX_WITH_VERIFY_LOCKS_BACKTRACE BOOL
  "Enable thread stack back trace being captured on lock registration (to be used in combination with HPX_WITH_VERIFY_LOCKS=ON, default: OFF)"
  OFF
  CATEGORY "Debugging" ADVANCED)
hpx_option(HPX_WITH_THREAD_DEBUG_INFO BOOL
  "Enable thread debugging information (default: OFF, implicitly enabled in debug builds)"
  OFF
  CATEGORY "Debugging" ADVANCED)
hpx_option(HPX_WITH_THREAD_GUARD_PAGE BOOL
  "Enable thread guard page (default: ON)"
  ON
  CATEGORY "Debugging" ADVANCED)

if(HPX_WITH_VERIFY_LOCKS)
  hpx_add_config_define(HPX_HAVE_VERIFY_LOCKS)
  if(HPX_WITH_VERIFY_LOCKS_BACKTRACE)
    hpx_add_config_define(HPX_HAVE_VERIFY_LOCKS_BACKTRACE)
  endif()
endif()
if(HPX_WITH_VERIFY_LOCKS_GLOBALLY)
  hpx_add_config_define(HPX_HAVE_VERIFY_LOCKS_GLOBALLY)
endif()

# Additional debug support
if(NOT WIN32 AND HPX_WITH_THREAD_GUARD_PAGE)
  hpx_add_config_define(HPX_HAVE_THREAD_GUARD_PAGE)
endif()

if(NOT WIN32 AND HPX_WITH_THREAD_STACK_MMAP)
  hpx_add_config_define(HPX_HAVE_THREAD_STACK_MMAP)
endif()

if(HPX_WITH_THREAD_MANAGER_IDLE_BACKOFF)
  hpx_add_config_define(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
endif()

hpx_option(HPX_WITH_THREAD_DESCRIPTION_FULL BOOL
  "Use function address for thread description (default: OFF)"
  OFF
  CATEGORY "Debugging" ADVANCED)

# If APEX is defined, the action timers need thread debug info.
if(HPX_WITH_APEX)
    hpx_add_config_define(HPX_HAVE_THREAD_DESCRIPTION)
    if(HPX_WITH_THREAD_DESCRIPTION_FULL)
      hpx_add_config_define(HPX_HAVE_THREAD_DESCRIPTION_FULL)
    endif()
endif()

if(HPX_WITH_THREAD_DEBUG_INFO)
  hpx_add_config_define(HPX_HAVE_THREAD_PARENT_REFERENCE)
  hpx_add_config_define(HPX_HAVE_THREAD_PHASE_INFORMATION)
  hpx_add_config_define(HPX_HAVE_THREAD_DESCRIPTION)
  hpx_add_config_define(HPX_HAVE_THREAD_DEADLOCK_DETECTION)
  if(HPX_WITH_THREAD_DESCRIPTION_FULL)
    hpx_add_config_define(HPX_HAVE_THREAD_DESCRIPTION_FULL)
  endif()
endif()

# run hpx_main on all localities by default
hpx_option(HPX_WITH_RUN_MAIN_EVERYWHERE BOOL "Run hpx_main by default on all localities (default: OFF)." OFF ADVANCED)
if(HPX_WITH_RUN_MAIN_EVERYWHERE)
  hpx_add_config_define(HPX_HAVE_RUN_MAIN_EVERYWHERE)
endif()

# Options for our plugins
hpx_option(HPX_WITH_COMPRESSION_BZIP2 BOOL
  "Enable bzip2 compression for parcel data (default: OFF)." OFF ADVANCED)
hpx_option(HPX_WITH_COMPRESSION_SNAPPY BOOL
  "Enable snappy compression for parcel data (default: OFF)." OFF ADVANCED)
hpx_option(HPX_WITH_COMPRESSION_ZLIB BOOL
  "Enable zlib compression for parcel data (default: OFF)." OFF ADVANCED)

# Parcel coalescing is used by the main HPX library, enable it always
hpx_option(HPX_WITH_PARCEL_COALESCING BOOL
  "Enable the parcel coalescing plugin (default: ON)." ON ADVANCED)
if(HPX_WITH_PARCEL_COALESCING)
  hpx_add_config_define(HPX_HAVE_PARCEL_COALESCING)
endif()

################################################################################
# Backwards compatibility options (edit for each release)

# HPX_WITH_COLOCATED_BACKWARDS_COMPATIBILITY: introduced in V0.9.11
hpx_option(HPX_WITH_COLOCATED_BACKWARDS_COMPATIBILITY BOOL
    "Enable backwards compatibility for apply_colocated, async_colocated and friends"
    ON ADVANCED)
if(HPX_WITH_COLOCATED_BACKWARDS_COMPATIBILITY)
  hpx_add_config_define(HPX_HAVE_COLOCATED_BACKWARDS_COMPATIBILITY)
endif()

# HPX_WITH_COMPONENT_GET_GID_COMPATIBILITY: introduced in V0.9.11
hpx_option(HPX_WITH_COMPONENT_GET_GID_COMPATIBILITY BOOL
    "Enable backwards compatibility for component::get_gid() functions"
    ON ADVANCED)
if(HPX_WITH_COMPONENT_GET_GID_COMPATIBILITY)
  hpx_add_config_define(HPX_HAVE_COMPONENT_GET_GID_COMPATIBILITY)
endif()

# HPX_WITH_LOCAL_DATAFLOW_COMPATIBILITY: introduced in V0.9.12
hpx_option(HPX_WITH_LOCAL_DATAFLOW_COMPATIBILITY BOOL
    "Enable backwards compatibility for hpx::lcos::local::dataflow() functions"
    ON ADVANCED)
if(HPX_WITH_LOCAL_DATAFLOW_COMPATIBILITY)
  hpx_add_config_define(HPX_HAVE_LOCAL_DATAFLOW_COMPATIBILITY)
endif()

# HPX_WITH_GENERIC_EXECUTION_POLICY: introduced in V0.9.12
hpx_option(HPX_WITH_GENERIC_EXECUTION_POLICY BOOL
    "Enable the generic execution policy (default: OFF)"
    OFF ADVANCED)
if(HPX_WITH_GENERIC_EXECUTION_POLICY)
  hpx_add_config_define(HPX_HAVE_GENERIC_EXECUTION_POLICY)
endif()

# BADBAD: This enables an overload of swap is necessary to work around the
#         problems caused by zip_iterator not being a real random access iterator.
#         Dereferencing zip_iterator does not yield a true reference but
#         only a temporary tuple holding true references.
#
# A real fix for this problem is proposed in PR0022R0
# (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0022r0.html)
#
hpx_option(HPX_WITH_TUPLE_RVALUE_SWAP BOOL
  "Enable swapping of rvalue tuples (needed for parallel::sort_by_key, default: ON)."
  ON CATEGORY "Utility" ADVANCED)
if(HPX_WITH_TUPLE_RVALUE_SWAP)
  hpx_add_config_define(HPX_HAVE_TUPLE_RVALUE_SWAP)
endif()

################################################################################
# Set basic search paths for HPX
################################################################################
include_directories("${PROJECT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")
link_directories(${CMAKE_BINARY_DIR}/lib)

################################################################################
# Check for compiler compatibility
#

# Check if the selected compiler versions are supposed to work with our codebase
if(CMAKE_COMPILER_IS_GNUCXX AND HPX_WITH_GCC_VERSION_CHECK)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.4.4)
    hpx_error("GCC 4.4.5 or higher is required. Specify HPX_GCC_VERSION_CHECK=OFF to ignore this error.")
  endif()
endif()

if(MSVC)
  if(NOT (MSVC11 OR MSVC12 OR MSVC14))
    hpx_error("MSVC x64 2012 or higher is required.")
  elseif(NOT CMAKE_CL_64)
    hpx_warn("MSVC (32Bit) will compile but will fail running larger applications because of limitations in the Windows OS.")
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
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    hpx_error("HPX on the BG/Q can only be compiled with bgclang")
  endif()
  hpx_info("Compiling for BlueGene/Q")
endif()

if(MSVC14 AND CMAKE_CL_64)
  hpx_option(HPX_WITH_AWAIT BOOL
      "Enable the use of experimental __await functionality"
      OFF ADVANCED CATEGORY "LCOs")
  if(HPX_WITH_AWAIT)
    hpx_add_config_define(HPX_HAVE_AWAIT)
    hpx_add_target_compile_option(-await)
  endif()
endif()

################################################################################

################################################################################
# Add necessary compiler flags. Flags added here include flags to disable/enable
# certain warnings, enabling C++11 mode and disabling asserts. Setting of
# optimization flags is not handled here and is left to the responsibility of
# the user to avoid conflicts in the resulting binaries

hpx_add_target_compile_definition(_DEBUG CONFIGURATIONS Debug)
hpx_add_target_compile_definition(DEBUG CONFIGURATIONS Debug)
hpx_add_target_compile_definition(HPX_DISABLE_ASSERTS
  CONFIGURATIONS Release RelWithDebInfo MinSizeRelease)
hpx_add_target_compile_definition(BOOST_DISABLE_ASSERTS
  CONFIGURATIONS Release RelWithDebInfo MinSizeRelease)

# Make sure we compile in C++11 mode (MSVC uses it automatically)
if(NOT MSVC)
  if(HPX_WITH_CUDA AND NOT HPX_WITH_CUDA_CLANG)
    hpx_add_target_compile_option(-std=c++11)
  else()
    # Try -std=c++14 first
    check_cxx_compiler_flag(-std=c++14 HPX_WITH_CXX14)
    if(HPX_WITH_CXX14)
      set(CXX_FLAG -std=c++14)
    else()
      # ... otherwise try -std=c++0y
      check_cxx_compiler_flag(-std=c++0y HPX_WITH_CXX0Y)
      if(HPX_WITH_CXX0Y)
        set(CXX_FLAG -std=c++0y)
      else()
        # ... otherwise try -std=c++11
        check_cxx_compiler_flag(-std=c++11 HPX_WITH_CXX11)
        if(HPX_WITH_CXX11)
          set(CXX_FLAG -std=c++11)
        else()
          # ... otherwise try -std=c++0x
          check_cxx_compiler_flag(-std=c++0x HPX_WITH_CXX0X)
          if(HPX_WITH_CXX0X)
            set(CXX_FLAG -std=c++0x)
          endif()
        endif()
      endif()
    endif()

    hpx_add_target_compile_option(${CXX_FLAG})
  endif()
endif()

################################################################################
# C++ feature tests
################################################################################

# Check the availability of certain C++11 language features
hpx_check_for_cxx11_alias_templates(
  DEFINITIONS HPX_HAVE_CXX11_ALIAS_TEMPLATES)

hpx_check_for_cxx11_auto(
  REQUIRED "HPX needs support for C++11 auto")

hpx_check_for_cxx11_constexpr(
  DEFINITIONS HPX_HAVE_CXX11_CONSTEXPR)

hpx_check_for_cxx11_decltype(
  REQUIRED "HPX needs support for C++11 decltype")
hpx_add_config_define(BOOST_RESULT_OF_USE_DECLTYPE)

hpx_check_for_cxx11_decltype_n3276(
  DEFINITIONS HPX_HAVE_CXX11_DECLTYPE_N3276)

hpx_check_for_cxx11_sfinae_expression(
  DEFINITIONS HPX_HAVE_CXX11_SFINAE_EXPRESSION)

hpx_check_for_cxx11_defaulted_functions(
  DEFINITIONS HPX_HAVE_CXX11_DEFAULTED_FUNCTIONS)

hpx_check_for_cxx11_deleted_functions(
  REQUIRED "HPX needs support for C++11 deleted functions")

hpx_check_for_cxx11_explicit_cvt_ops(
  REQUIRED "HPX needs support for C++11 explicit conversion operators")

hpx_check_for_cxx11_function_template_default_args(
  REQUIRED "HPX needs support for C++11 defaulted function template arguments")

hpx_check_for_cxx11_initializer_list(
  DEFINITIONS HPX_HAVE_CXX11_INITIALIZER_LIST)

hpx_check_for_cxx11_inline_namespaces(
  DEFINITIONS HPX_HAVE_CXX11_INLINE_NAMESPACES)

hpx_check_for_cxx11_lambdas(
  DEFINITIONS HPX_HAVE_CXX11_LAMBDAS)

hpx_check_for_cxx11_noexcept(
  DEFINITIONS HPX_HAVE_CXX11_NOEXCEPT)

hpx_check_for_cxx11_nullptr(
  DEFINITIONS HPX_HAVE_CXX11_NULLPTR)

hpx_check_for_cxx11_range_based_for(
  REQUIRED "HPX needs support for C++11 range-based for-loop")

hpx_check_for_cxx11_rvalue_references(
  REQUIRED "HPX needs support for C++11 rvalue references")

hpx_check_for_cxx11_scoped_enums(
  REQUIRED "HPX needs support for C++11 scoped enums")

hpx_check_for_cxx11_static_assert(
  REQUIRED "HPX needs support for C++11 static_assert")

hpx_check_for_cxx11_variadic_macros(
  REQUIRED "HPX needs support for C++11 variadic macros")

hpx_check_for_cxx11_variadic_templates(
  REQUIRED "HPX needs support for C++11 variadic templates")

hpx_check_for_cxx11_explicit_variadic_templates(
  DEFINITIONS HPX_HAVE_CXX11_EXPLICIT_VARIADIC_TEMPLATES)

hpx_check_for_cxx11_extended_friend_declarations(
  DEFINITIONS HPX_HAVE_CXX11_EXTENDED_FRIEND_DECLARATIONS)

# Check the availability of certain C++11 library features
hpx_check_for_cxx11_std_array(
  DEFINITIONS HPX_HAVE_CXX11_STD_ARRAY)

hpx_check_for_cxx11_std_chrono(
  DEFINITIONS HPX_HAVE_CXX11_STD_CHRONO)

hpx_check_for_cxx11_std_cstdint(
  DEFINITIONS HPX_HAVE_CXX11_STD_CSTDINT)

hpx_check_for_cxx11_std_initializer_list(
  DEFINITIONS HPX_HAVE_CXX11_STD_INITIALIZER_LIST)

hpx_check_for_cxx11_std_is_bind_expression(
  DEFINITIONS HPX_HAVE_CXX11_STD_IS_BIND_EXPRESSION)

hpx_check_for_cxx11_std_is_final(
  DEFINITIONS HPX_HAVE_CXX11_STD_IS_FINAL)

hpx_check_for_cxx11_std_is_null_pointer(
  DEFINITIONS HPX_HAVE_CXX11_STD_IS_NULL_POINTER)

hpx_check_for_cxx11_std_is_placeholder(
  DEFINITIONS HPX_HAVE_CXX11_STD_IS_PLACEHOLDER)

hpx_check_for_cxx11_std_lock_guard(
  REQUIRED "HPX needs support for C++11 std::lock_guard")

hpx_check_for_cxx11_std_reference_wrapper(
  DEFINITIONS HPX_HAVE_CXX11_STD_REFERENCE_WRAPPER)

hpx_check_for_cxx11_std_shared_ptr(
  REQUIRED "HPX needs support for C++11 std::shared_ptr")

hpx_check_for_cxx11_std_to_string(
  REQUIRED "HPX needs support for C++11 std::to_string")

hpx_check_for_cxx11_std_unique_lock(
  REQUIRED "HPX needs support for C++11 std::unique_lock")

hpx_check_for_cxx11_std_unique_ptr(
  REQUIRED "HPX needs support for C++11 std::unique_ptr")

hpx_check_for_cxx11_std_unordered_map(
  REQUIRED "HPX needs support for C++11 std::unordered_map")

hpx_check_for_cxx11_std_unordered_set(
  REQUIRED "HPX needs support for C++11 std::unordered_set")

hpx_check_for_cxx11_std_type_traits(
  DEFINITIONS HPX_HAVE_CXX11_STD_TYPE_TRAITS)

hpx_check_for_cxx11_is_trivially_copyable(
  DEFINITIONS HPX_HAVE_CXX11_IS_TRIVIALLY_COPYABLE)

# Check the availability of certain C++14 language features
hpx_check_for_cxx14_constexpr(
  DEFINITIONS HPX_HAVE_CXX14_CONSTEXPR)

hpx_check_for_cxx14_lambdas(
  DEFINITIONS HPX_HAVE_CXX14_LAMBDAS)

# Check the availability of certain C++14 library features
hpx_check_for_cxx14_std_integer_sequence(
  DEFINITIONS HPX_HAVE_CXX14_STD_INTEGER_SEQUENCE)

hpx_check_for_cxx14_std_result_of_sfinae(
  DEFINITIONS HPX_HAVE_CXX14_STD_RESULT_OF_SFINAE)

# check for experimental facilities
hpx_check_for_cxx_experimental_std_optional(
  DEFINITIONS HPX_HAVE_CXX1Y_EXPERIMENTAL_OPTIONAL)

################################################################################
# CUDA features
################################################################################
if(HPX_WITH_CUDA AND NOT HPX_WITH_CUDA_CLANG)
  hpx_libraries(cudadevrt)
  if(NOT MSVC)
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
#     set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-D_DEBUG;-O0;-g;-std=c++11;-Xcompiler=-g,-O0)
#     set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO};-DNDEBUG;-O3;-std=c++11;-g;-Xcompiler=-g,-O3)
#     set(CUDA_NVCC_FLAGS_MINSIZEREL ${CUDA_NVCC_FLAGS_MINSIZEREL};-DNDEBUG;-O1-std=c++11;;-Xcompiler=-g,-O1)
#     set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};-DNDEBUG;-O3;-std=c++11;-Xcompiler=-O3)
  else()
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-D_DEBUG;-O0;-g;--keep;-Xcompiler=-MDd,-Od,-Zi)
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO};-DNDEBUG;-O2;-g;-Xcompiler=-MD,-O2,-Zi)
    set(CUDA_NVCC_FLAGS_MINSIZEREL ${CUDA_NVCC_FLAGS_MINSIZEREL};-DNDEBUG;-O1;-Xcompiler=-MD,-O1)
    set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};-DNDEBUG;-O2;-Xcompiler=-MD,-Ox)
  endif()
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_30,code=sm_30)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_35,code=sm_35)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_30,code=compute_30)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_35,code=compute_35)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-extended-lambda)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lcudadevrt)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-rdc=true)
endif()
if(HPX_WITH_CUDA AND HPX_WITH_CUDA_CLANG)
  hpx_add_target_compile_option(--cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
  hpx_libraries(cudart)
  link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif()

################################################################################
# Check for misc system headers
################################################################################

hpx_check_for_unistd_h(
  DEFINITIONS HPX_HAVE_UNISTD_H)

if(NOT WIN32)
  ##############################################################################
  # Macro definitions for system headers
  ##############################################################################
  add_definitions(-D_GNU_SOURCE)
  if(APPLE)
    hpx_add_config_define(BOOST_HAS_INT128)
  endif()

  ##############################################################################
  # System libraries
  ##############################################################################
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
    hpx_libraries(dl)
  endif()

  if(NOT APPLE AND NOT ("${HPX_PLATFORM_UC}" STREQUAL "ANDROID"))
    hpx_libraries(rt)
  endif()

  if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
    hpx_libraries(log)
  endif()

  if(APPLE)
    hpx_add_compile_flag_if_available(-ftemplate-depth=256 LANGUAGES CXX)
  endif()
endif()

if(WIN32)
  if(MSVC)
    enable_language(ASM_MASM)
    hpx_add_target_compile_option(-Ox CONFIGURATIONS Release)

    if(NOT HPX_WITH_AWAIT)
      # /RTC1 is incompatible with /await
      hpx_add_target_compile_option(/RTC1 CONFIGURATIONS Debug)
    else()
      hpx_remove_target_compile_option(/RTC1 CONFIGURATIONS Debug)
    endif()

    # VS2012 and above has a special flag for improving the debug experience by
    # adding more symbol information to the build (-d2Zi)
    hpx_add_target_compile_option(-d2Zi+ CONFIGURATIONS RelWithDebInfo)

    # VS2013 and above know how to do link time constant data segment folding
    # VS2013 update 2 and above know how to remove debug information for
    #     non-referenced functions and data (-Zc:inline)
    if(MSVC12 OR MSVC14)
      hpx_add_target_compile_option(-Zc:inline)
      hpx_add_target_compile_option(-Gw
        CONFIGURATIONS Release RelWithDebInfo MinSizeRelease)
      hpx_add_target_compile_option(-Zo CONFIGURATIONS RelWithDebInfo)
    endif()

    # Exceptions
    hpx_add_target_compile_option(-EHsc)
    if(MSVC14)
      # assume conforming (throwing) operator new implementations
      hpx_add_target_compile_option(-Zc:throwingNew)

      # enable faster linking (requires VS2015 Update1)
      # disabled for now as this flag crashes debugger
      # hpx_remove_link_flag(/debug CONFIGURATIONS Debug)
      # hpx_add_link_flag(/debug:fastlink CONFIGURATIONS Debug)

      # Update 2 requires to set _ENABLE_ATOMIC_ALIGNMENT_FIX for it to compile
      # atomics
      hpx_add_config_define(_ENABLE_ATOMIC_ALIGNMENT_FIX)
    endif()

    # Runtime type information
    hpx_add_target_compile_option(-GR)
    # Multiprocessor build
    hpx_add_target_compile_option(-MP)
    # Increase the maximum size of object file sections
    hpx_add_target_compile_option(-bigobj)
  endif()

  hpx_libraries(psapi shlwapi)

  ##############################################################################
  # Macro definitions for system headers
  ##############################################################################
  add_definitions(-D_WINDOWS)
  add_definitions(-D_WIN32)
  hpx_add_config_define(_WIN32_WINNT 0x0601)
  hpx_add_config_define(_SCL_SECURE_NO_WARNINGS)
  hpx_add_config_define(_CRT_SECURE_NO_WARNINGS)
  hpx_add_config_define(_SCL_SECURE_NO_DEPRECATE)
  hpx_add_config_define(_CRT_SECURE_NO_DEPRECATE)
  hpx_add_config_define(_CRT_NONSTDC_NO_WARNINGS)
  hpx_add_config_define(_WINSOCK_DEPRECATED_NO_WARNINGS)

  ##############################################################################
  # Boost
  ##############################################################################

  hpx_add_config_define(BOOST_USE_WINDOWS_H)
  hpx_add_config_define(BOOST_SERIALIZATION_DYN_LINK)
  if(NOT CMAKE_CL_64)
    hpx_add_config_define(BOOST_NO_ALIGNMENT)
  endif()
  if(NOT HPX_WITH_GENERIC_CONTEXT_COROUTINES)
    hpx_add_config_define(HPX_HAVE_FIBER_BASED_COROUTINES)
  endif()
  hpx_add_config_define(PSAPI_VERSION 1)
endif()

# Configure Warnings
if(HPX_WITH_COMPILER_WARNINGS)
  if(MSVC) # Adding special warning settings for the MSVC compiler ...
    hpx_add_compile_flag(-W3 LANGUAGES C CXX)
    # According to the ifort Windows manual, W3 isn't supported
    hpx_add_compile_flag(-W1 LANGUAGES Fortran)
    # Boost.Lockfree triggers 'warning C4307: '+' : integral constant overflow'
    # which is benign
    hpx_add_compile_flag(-wd4307)

    # MSVC2012/2013 are overeager to report 'qualifier applied to function type
    # has no meaning; ignored'
    hpx_add_compile_flag(-wd4180)

    # max symbol length exceeded
    hpx_add_compile_flag(-wd4503)

    # OpenMPI triggers a lot of forcing value to bool 'true' or 'false'
    # (performance warning)
    if(MPI_FOUND OR HPX_WITH_APEX)
      hpx_add_compile_flag(-wd4800)
    endif()
  else() # Trial and error approach for any other compiler ...
    hpx_add_compile_flag_if_available(-Wall LANGUAGES CXX C Fortran)
    hpx_add_compile_flag_if_available(-Wextra LANGUAGES CXX C Fortran)
    # This is a new warning popping up from the boost headers with no particular meaning
    hpx_add_compile_flag_if_available(-Wno-unused-local-typedefs LANGUAGES CXX C Fortran)
    hpx_add_compile_flag_if_available(-Wno-strict-aliasing LANGUAGES CXX C Fortran)
    hpx_add_compile_flag_if_available(-Wno-sign-promo LANGUAGES CXX)
    hpx_add_compile_flag_if_available(-Wno-attributes LANGUAGES CXX)
    hpx_add_compile_flag_if_available(-Wno-cast-align LANGUAGES CXX)

    # These are usually benign and can't be suppressed because of
    # interface requirements
    hpx_add_compile_flag_if_available(-Wno-unused-parameter)

    # Be extra strict about format checks
    # Boost.Logging is built on fprintf, sadly
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

    hpx_add_compile_flag_if_available(-Werror=trampolines)
    hpx_add_compile_flag_if_available(-Werror=parentheses)
    hpx_add_compile_flag_if_available(-Werror=reorder)
    hpx_add_compile_flag_if_available(-Werror=return-type)
    hpx_add_compile_flag_if_available(-Werror=sequence-point)

    if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
      # Uninitialized variables are bad, earlier compilers issue spurious
      # warnings
      hpx_add_compile_flag_if_available(-Werror=uninitialized)
    endif()

    hpx_add_compile_flag_if_available(-Werror=format)
    hpx_add_compile_flag_if_available(-Werror=missing-braces)
    hpx_add_compile_flag_if_available(-Werror=sign-compare)
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
    hpx_add_compile_flag_if_available(-Wno-unused-local-typedefs LANGUAGES CXX C)
  endif()

  # Silence warning about __sync_fetch_and_nand changing semantics
  hpx_add_compile_flag_if_available(-Wno-sync-nand LANGUAGES CXX C)

  # Silence warnings about deleting polymorphic objects with non-virtual dtors.
  # These come from within Boost. Detection for this flag fails with GCC 4.4,
  # 4.5 and 4.6.
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.6)
    hpx_add_compile_flag_if_available(-Wno-delete-non-virtual-dtor LANGUAGES CXX)
  endif()

  # Check if our libraries have unresolved symbols
  #if(NOT APPLE AND NOT HPX_WITH_APEX)
  if(NOT APPLE AND NOT WIN32 AND NOT HPX_WITH_SANITIZERS)
    hpx_add_link_flag_if_available(-Wl,-z,defs TARGETS SHARED EXE)
  endif()
  if(WIN32)
    hpx_libraries(WS2_32)
    hpx_libraries(mswsock)
  endif()

  if("${HPX_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
    hpx_add_compile_flag_if_available(-Wno-deprecated-register LANGUAGES CXX C)
  endif()

  if(HPX_WITH_HIDDEN_VISIBILITY)
    hpx_add_compile_flag_if_available(-fvisibility=hidden LANGUAGES CXX C Fortran)
    hpx_add_link_flag_if_available(-fvisibility=hidden TARGETS SHARED EXE)
    hpx_add_config_define(HPX_HAVE_ELF_HIDDEN_VISIBILITY)
    hpx_add_config_define(HPX_HAVE_COROUTINE_GCC_HIDDEN_VISIBILITY)
    hpx_add_config_define(HPX_HAVE_PLUGIN_GCC_HIDDEN_VISIBILITY)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    hpx_add_compile_flag_if_available(-Wno-cast-align)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # Disable the following warnings:
    # #1170: invalid redeclaration of nested class
    hpx_add_compile_flag_if_available(-wd1170)
    # #858: type qualifier on return type is meaningless
    hpx_add_compile_flag_if_available(-wd858)
    # #1098: the qualifier on this friend declaration is ignored
    hpx_add_compile_flag_if_available(-wd1098)
    # #488: template parameter not used in declaring the parameter type
    hpx_add_compile_flag_if_available(-wd488)
    # #2203: cast discards qualifiers from target type (needed for mvapich2
    #        mpi header)
    hpx_add_compile_flag_if_available(-wd2203)
    # #2536: cannot specify explicit initializer for arrays
    hpx_add_compile_flag_if_available(-wd2536)
  endif()

  # rdtsc is an x86 instruction that reads the value of a CPU time stamp
  # counter. rdtscp is an extension to rdtsc. The difference is that rdtscp is
  # a serializing instruction.
  hpx_cpuid("rdtsc" HPX_WITH_RDTSC
    DEFINITIONS HPX_HAVE_RDTSC)

  # XeonPhi's do not support RDTSCP
  if(NOT ("${HPX_PLATFORM_UC}" STREQUAL "XEONPHI"))
    hpx_cpuid("rdtscp" HPX_WITH_RDTSCP
      DEFINITIONS HPX_HAVE_RDTSCP)
  endif()

  if(NOT HPX_WITH_RDTSC AND NOT HPX_WITH_RDTSCP)
    hpx_warn("Neither rdtsc nor rdtscp is available; some performance counters may report incorrect results")
  endif()
endif()

################################################################################

################################################################################
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

# convey selected allocator type to the build configuration
hpx_add_config_define(HPX_HAVE_MALLOC "\"${HPX_WITH_MALLOC}\"")

if(HPX_WITH_HWLOC)
  find_package(Hwloc)
  if(NOT HWLOC_FOUND)
    hpx_error("Hwloc could not be found and HPX_WITH_HWLOC=ON, please specify HWLOC_ROOT to point to the correct location or set HPX_WITH_HWLOC to OFF")
  endif()
  hpx_libraries(${HWLOC_LIBRARIES})
  include_directories(${HWLOC_INCLUDE_DIR})
  hpx_add_config_define(HPX_HAVE_HWLOC)
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

if(HPX_WITH_TAU)
  find_package(TAU)
  if(NOT TAU_FOUND)
    hpx_error("TAU could not be found and HPX_WITH_TAU=On, please specify TAU_ROOT to point to the root of your TAU installation")
  endif()
  hpx_libraries(${TAU_LIBRARIES})
  include_directories(${TAU_INCLUDE_DIR})
endif()

################################################################################
# Enable integration with Apex event counters
################################################################################
if(HPX_WITH_APEX)
  # handle APEX library
  include(GitExternal)
  git_external(apex
    https://github.com/khuck/xpress-apex.git
    v0.5
    VERBOSE)

  LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/apex/cmake/Modules")
  add_subdirectory(apex/src/apex)
  if(NOT APEX_FOUND)
    hpx_error("Apex could not be found and HPX_WITH_APEX=On")
  endif()
  if(AMPLIFIER_FOUND)
    hpx_error("AMPLIFIER_FOUND has been set. Please disable the use of the Intel Amplifier (WITH_AMPLIFIER=Off) in order to use Apex")
  endif()

  include_directories(${APEX_SOURCE_DIR})
  set(HPX_ADDITIONAL_RUNTIME_DEPENDENCIES
      ${HPX_ADDITIONAL_RUNTIME_DEPENDENCIES} apex_lib)
  if(APEX_WITH_MSR)
    set(HPX_ADDITIONAL_RUNTIME_DEPENDENCIES
        ${HPX_ADDITIONAL_RUNTIME_DEPENDENCIES} -L${MSR_ROOT}/lib -lmsr)
  endif()
  if(APEX_WITH_ACTIVEHARMONY)
    set(HPX_ADDITIONAL_RUNTIME_DEPENDENCIES
        ${HPX_ADDITIONAL_RUNTIME_DEPENDENCIES} -L${ACTIVEHARMONY_ROOT}/lib -lharmony)
  endif()
  if(APEX_WITH_TAU OR HPX_WITH_TAU)
    set(HPX_ADDITIONAL_RUNTIME_DEPENDENCIES
        ${HPX_ADDITIONAL_RUNTIME_DEPENDENCIES} -L${TAU_ROOT}/lib -lTAU)
  endif()

  # handle optional ITTNotify library
  if(HPX_WITH_ITTNOTIFY)
    add_subdirectory(apex/src/ITTNotify)
    if(NOT ITTNOTIFY_FOUND)
      hpx_error("ITTNotify could not be found and HPX_WITH_ITTNOTIFY=On")
    endif()
    include_directories(${ITTNOTIFY_SOURCE_DIR})
    hpx_add_config_define(HPX_HAVE_ITTNOTIFY 1)
    set(HPX_ADDITIONAL_RUNTIME_DEPENDENCIES
        ${HPX_ADDITIONAL_RUNTIME_DEPENDENCIES} ittnotify_lib)
  endif()
endif()

if(HPX_WITH_GOOGLE_PERFTOOLS)
  find_package(GooglePerftools)
  if(NOT GOOGLE_PERFTOOLS_FOUND)
    hpx_error("Google Perftools could not be found and HPX_WITH_GOOGLE_PERFTOOLS=On, please specify GOOGLE_PERFTOOLS to point to the root of your Google Perftools installation")
  endif()
  hpx_libraries(${GOOGLE_PERFTOOLS_LIBRARIES})
  include_directories(${GOOGLE_PERFTOOLS_INCLUDE_DIR})
endif()

if(HPX_WITH_VALGRIND)
  find_package(Valgrind)
  if(NOT VALGRIND_FOUND)
    hpx_error("Valgrind could not be found and HPX_WITH_VALGRIND=On, please specify VALGRIND_ROOT to point to the root of your Valgrind installation")
  endif()
  include_directories(${VALGRIND_INCLUDE_DIR})
  hpx_add_config_define(HPX_HAVE_VALGRIND)
endif()

################################################################################
# Security, libsodium: P(ortable|ackageable) NaCl
################################################################################
if(HPX_WITH_SECURITY)
  find_package(Sodium)
  if(NOT SODIUM_FOUND)
    hpx_error("Sodium could not be found and WITH_SODIUM=On, please specify Sodium_ROOT to point to the root of your Sodium installation")
  endif()
  hpx_add_config_define(HPX_HAVE_SODIUM)

  include_directories(${SODIUM_INCLUDE_DIR})

  if(MSVC)
    hpx_add_config_define(SODIUM_STATIC)
  endif()
  hpx_libraries(${SODIUM_LIBRARIES})

  hpx_add_config_define(HPX_HAVE_SECURITY)
endif()
################################################################################

################################################################################
# Check Build Options based on the found dependencies. We also check for errors
# with incompatible options with the currently selected platform.
#

if(HPX_WITH_GENERIC_CONTEXT_COROUTINES)
  # Check if we can use generic coroutine contexts without any problems
  if(NOT Boost_CONTEXT_FOUND)
    hpx_error("The usage of Boost.Context was selected but Boost.Context was not found (Version 1.51 or higher is required).")
  endif()
  if("${HPX_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
    if(Boost_VERSION LESS 105600)
      hpx_error("On BlueGene/Q, Boost.Context can only be used with a Boost >=1.56")
    endif()
  endif()
  hpx_add_config_define(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
endif()

################################################################################
# Emulation of SwapContext on Windows
################################################################################
if(WIN32)
  if(HPX_WITH_SWAP_CONTEXT_EMULATION)
    if(NOT CMAKE_ASM_MASM_COMPILER)
      hpx_error("SwitchToFiber emulation can not be enabled. The masm compiler could not be found. Try setting the ASM_MASM environment variable to the assembler executable (ml.exe/ml64.exe) or disable the emulation by setting HPX_WITH_SWAP_CONTEXT_EMULATION to Off")
    else()
      hpx_info("SwitchToFiber emulation is enabled, using compiler: '${CMAKE_ASM_MASM_COMPILER}'")
      hpx_add_config_define(HPX_HAVE_SWAP_CONTEXT_EMULATION)
    endif()
  endif()
endif()

################################################################################

################################################################################
# HPX_PREFIX
# The prefix is the default search path for HPX plugins
################################################################################
if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
  set(HPX_PREFIX "lib")
  set(HPX_BUILD_PREFIX "lib")
else()
  set(HPX_PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(HPX_BUILD_PREFIX "${CMAKE_BINARY_DIR}")
endif()


# ################################################################################
# # search path configuration
# ################################################################################

if (NOT DEFINED LIB)
  set(LIB "lib")
endif(NOT DEFINED LIB)

if(HPX_WITH_FULL_RPATH)
  hpx_include(SetFullRPATH)
endif()
###############################################################################

################################################################################
# Git commit detection
################################################################################
include(HPX_GitCommit)
hpx_add_config_define(HPX_HAVE_GIT_COMMIT "\"${HPX_WITH_GIT_COMMIT}\"")

hpx_include(SetOutputPaths)
##############################################################################

################################################################################
# Configure compression and other plugins
################################################################################
set(HPX_STATIC_PARCELPORT_PLUGINS "" CACHE INTERNAL "" FORCE)
add_hpx_pseudo_target(plugins)
add_subdirectory(plugins)

if(HPX_WITH_COMPRESSION_BZIP2)
  hpx_add_config_define(HPX_HAVE_COMPRESSION_BZIP2)
endif()
if(HPX_WITH_COMPRESSION_SNAPPY)
  hpx_add_config_define(HPX_HAVE_COMPRESSION_SNAPPY)
endif()
if(HPX_WITH_COMPRESSION_ZLIB)
  hpx_add_config_define(HPX_HAVE_COMPRESSION_ZLIB)
endif()

################################################################################
# Documentation toolchain (DocBook, BoostBook, QuickBook, xsltproc)
################################################################################
hpx_include(Documentation)

# Debug library postfix
set(CMAKE_DEBUG_POSTFIX "d")
set(HPX_DEBUG_POSTFIX "d")

################################################################################
# Target specification
################################################################################
# Recurse into some subdirectories. This does not actually cause another cmake
# executable to run. The same process will walk through the project's entire
# directory structure.
if(HPX_WITH_TOOLS)
  add_hpx_pseudo_target(tools)
  add_subdirectory(tools)
endif()

add_hpx_pseudo_target(core)
add_subdirectory(src)

if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples)
  include_directories(examples)
  add_subdirectory(examples)
endif()

###############################################################################
# Tests
###############################################################################
if(HPX_WITH_TESTS)
  find_package(PythonInterp)

  if(NOT PYTHONINTERP_FOUND)
    hpx_warn("A python interpreter could not be found. The test suite can not be run automatically.")
  endif()

  add_hpx_pseudo_target(tests)

  enable_testing()
  include(CTest)

  if(NOT MSVC)
    add_custom_command(TARGET tests POST_BUILD
      COMMAND ctest --output-on-failure --timeout 100)
  endif()

  include_directories(tests)
  add_subdirectory(tests)
endif()

if(HPX_WITH_RUNTIME)
  add_hpx_pseudo_target(runtime)
  add_subdirectory(runtime)
endif()

if(HPX_WITH_DOCUMENTATION)
  add_subdirectory(docs)
endif()

###############################################################################
# Activate plugins
###############################################################################
add_plugin_modules()

################################################################################
# Configure the header to include all compile definitions
################################################################################
get_property(HPX_CONFIG_DEFINITIONS_VAR GLOBAL PROPERTY HPX_CONFIG_DEFINITIONS)

list(SORT HPX_CONFIG_DEFINITIONS_VAR)
list(REMOVE_DUPLICATES HPX_CONFIG_DEFINITIONS_VAR)

set(hpx_config_defines "\n")
foreach(def ${HPX_CONFIG_DEFINITIONS_VAR})
  set(hpx_config_defines "${hpx_config_defines}#define ${def} ${${def}_define}\n")#"
endforeach()

# Generate a defines.hpp to be used in the build directory ...
set(HPX_DEFINES_PREFIX ${HPX_BUILD_PREFIX})
configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/config_defines.hpp.in"
               "${CMAKE_BINARY_DIR}/hpx/config/defines.hpp"
               @ONLY)

# Generate a defines.hpp to be used in the install directory ...
set(HPX_DEFINES_PREFIX ${HPX_PREFIX})
configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/config_defines.hpp.in"
               "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpx/config/defines.hpp"
               @ONLY)

# Configure hpxrun.py
configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/hpxrun.py.in"
               "${CMAKE_BINARY_DIR}/bin/hpxrun.py"
               @ONLY)

################################################################################
# installation instructions
################################################################################
install(
  FILES "${CMAKE_BINARY_DIR}/bin/hpxrun.py"
  DESTINATION bin
  COMPONENT core
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
  GROUP_READ GROUP_EXECUTE
  WORLD_READ WORLD_EXECUTE)

install( # install all hpx header files
  DIRECTORY hpx/
  DESTINATION include/hpx
  COMPONENT core
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE)

# Install all HPX header that have been configured using various
# cmake options
install(
  DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpx/"
  DESTINATION include/hpx
  COMPONENT core
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE)

install( # install all hpx header files
  DIRECTORY hpx
  DESTINATION share/hpx-${HPX_VERSION}/docs/html/code
  COMPONENT core
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE)

install( # install all hpx source files
  DIRECTORY src
  DESTINATION share/hpx-${HPX_VERSION}/docs/html/code
  COMPONENT core
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE)

install( # install all hpx examples source files
  DIRECTORY examples
  DESTINATION share/hpx-${HPX_VERSION}/docs/html/code
  COMPONENT core
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE)

install( # install all hpx tests source files
  DIRECTORY tests
  DESTINATION share/hpx-${HPX_VERSION}/docs/html/code
  COMPONENT core
  PATTERN ".git" EXCLUDE
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "CTestFiles" EXCLUDE)

install( # Install all HPX cmake utility files
  DIRECTORY cmake/
  DESTINATION ${LIB}/cmake/${HPX_PACKAGE_NAME}
  COMPONENT core
  PATTERN ".cmake.in" EXCLUDE
  PATTERN ".git" EXCLUDE)

install( # Install HPX Python scripts
  DIRECTORY python/scripts/
  DESTINATION bin
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                   GROUP_READ GROUP_EXECUTE
                   WORLD_READ WORLD_EXECUTE
  COMPONENT core
  FILES_MATCHING PATTERN "*.py"
  PATTERN ".git" EXCLUDE)

#   if(UNIX)
#     file(GLOB scripts "${PROJECT_SOURCE_DIR}/python/scripts/*.py")
#     execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory
#       "${CMAKE_BINARY_DIR}/python/scripts" )
#     foreach(script ${scripts})
#       get_filename_component(script_name ${script} NAME)
#       get_filename_component(script_name_we ${script} NAME_WE)
#
#       #make copy, so that we have intact symlink in CMAKE_BINARY_DIR
#       execute_process(
#         COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${script}" "${script_name}"
#   WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python/scripts")
#
#       execute_process(
#         COMMAND "${CMAKE_COMMAND}" -E create_symlink "${script_name}" "${script_name_we}"
#   WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python/scripts")
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
#   execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink "${PROJECT_SOURCE_DIR}/python/hpx"
#     "hpx-${HPX_VERSION}" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python")
#   # create a symlink in share pointing to the latest HPX installation
#   execute_process(COMMAND "${CMAKE_COMMAND}" -E create_symlink "hpx-${HPX_VERSION}" "hpx" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python")
#   install(DIRECTORY "${CMAKE_BINARY_DIR}/python/hpx" DESTINATION "share")

if("${HPX_PLATFORM_UC}" STREQUAL "XEONPHI")
  # FIXME: push changes upstream
  install(
    DIRECTORY external/asio/boost
    DESTINATION include/hpx/external
    COMPONENT core
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".git" EXCLUDE)
endif()

if(Boost_VERSION LESS 105300)
  # Install Boost.Atomic library for older Boost versions only
  install(
    DIRECTORY external/atomic/boost
    DESTINATION include/hpx/external
    COMPONENT core
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".git" EXCLUDE)

  # Install Boost.Lockfree library for older Boost versions only
  install(
    DIRECTORY external/lockfree/boost
    DESTINATION include/hpx/external
    COMPONENT core
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".git" EXCLUDE)
endif()

install(
  FILES "${PROJECT_SOURCE_DIR}/LICENSE_1_0.txt"
  DESTINATION share/hpx-${HPX_VERSION}
  COMPONENT license)

if(HPX_WITH_DOCUMENTATION)
  # Install HPX documentation files
  install(
    DIRECTORY "${CMAKE_BINARY_DIR}/share/hpx-${HPX_VERSION}/docs/"
    DESTINATION share/hpx-${HPX_VERSION}/docs
    COMPONENT docs
    OPTIONAL
    FILES_MATCHING PATTERN "*.html"
    PATTERN "*.pdf"
    PATTERN ".git" EXCLUDE
    PATTERN "*code*" EXCLUDE
    PATTERN "*src*" EXCLUDE
    PATTERN "*images*" EXCLUDE)

  install(
    FILES "${PROJECT_SOURCE_DIR}/docs/index.html"
    DESTINATION share/hpx-${HPX_VERSION}/
    COMPONENT docs)

  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/docs/html/src/"
    DESTINATION share/hpx-${HPX_VERSION}/docs/html/src
    COMPONENT docs
    FILES_MATCHING PATTERN "*.css"
    PATTERN ".git" EXCLUDE)

  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/docs/html/images/"
    DESTINATION share/hpx-${HPX_VERSION}/docs/html/images
    COMPONENT docs
    FILES_MATCHING PATTERN "*.png"
    PATTERN ".git" EXCLUDE)
endif()

################################################################################
# External build system support (FindHPX.cmake and pkg-config).
################################################################################

include(HPX_GeneratePackage)

# Create a symlink in share pointing to the latest HPX installation
execute_process(COMMAND
  "${CMAKE_COMMAND}" -E create_symlink "hpx-${HPX_VERSION}" "hpx"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/share")

message("")
message("HPX will be installed to ${CMAKE_INSTALL_PREFIX}")
message("")
