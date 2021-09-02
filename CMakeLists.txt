# Copyright (c)      2020 Mikael Simberg
# Copyright (c) 2007-2020 Hartmut Kaiser
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
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HPXLocal_CXXOverrides.cmake"
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
project(HPXLocal CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ##############################################################################
# Special handling for building tests/examples against a previously installed
# version of HPX
# ##############################################################################
if(HPXLocal_WITH_INSTALLED_VERSION)
  include(cmake/installed_hpx.cmake)
  return()
endif()

# ##############################################################################
# Build all of HPX
# ##############################################################################
set(HPXLocal_VERSION_MAJOR 0)
set(HPXLocal_VERSION_MINOR 1)
set(HPXLocal_VERSION_SUBMINOR 0)
set(HPXLocal_VERSION_DATE 20210714)
set(HPXLocal_VERSION_TAG "-trunk")

set(HPXLocal_VERSION
    "${HPXLocal_VERSION_MAJOR}.${HPXLocal_VERSION_MINOR}.${HPXLocal_VERSION_SUBMINOR}"
)
set(HPXLocal_LIBRARY_VERSION "${HPXLocal_VERSION}")
set(HPXLocal_SOVERSION ${HPXLocal_VERSION_MAJOR})
set(HPXLocal_PACKAGE_NAME HPXLocal)
# To keep track of the hpx_local_root when other subprojects are declared
set(HPXLocal_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
set(HPXLocal_BINARY_DIR "${PROJECT_BINARY_DIR}")

# ##############################################################################
# CMake configuration
# ##############################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(HPXLocal_Utils)

# explicitly set certain policies
cmake_policy(VERSION 3.18)
hpx_local_set_cmake_policy(CMP0042 NEW)
hpx_local_set_cmake_policy(CMP0060 NEW)
hpx_local_set_cmake_policy(CMP0074 NEW)

# We save the passed compiler flag to a special variable. This is needed for our
# build system unit tests. Some flags might influence the created symbols
# (_GLIBCXX_DEBUG i look at you)
set(CMAKE_CXX_FLAGS_SAFE ${CMAKE_CXX_FLAGS})
include(HPXLocal_CheckCXXStandard)

include(CheckCXXCompilerFlag)
include(CMakeDependentOption)

# include additional macro definitions
include(HPXLocal_AddConfigTest)
include(HPXLocal_AddDefinitions)
include(HPXLocal_CreateSymbolicLink)

hpx_local_force_out_of_tree_build(
  "This project requires an out-of-source-tree build. See README.rst. Clean your CMake cache and CMakeFiles if this message persists."
)

if(NOT HPXLocal_CMAKE_LOGLEVEL)
  set(HPXLocal_CMAKE_LOGLEVEL "WARN")
endif()

# print initial diagnostics
hpx_local_info("CMake version: ${CMAKE_VERSION}")
hpx_local_info("HPXLocal version: ${HPXLocal_VERSION}")

# ##############################################################################
# reset cached variables that need to be re-filled
unset(HPXLocal_EXPORT_TARGETS CACHE)
unset(HPXLocal_EXPORT_INTERNAL_TARGETS CACHE)
unset(HPXLocal_LIBS CACHE)
unset(HPXLocal_ENABLED_MODULES CACHE)

# ##############################################################################
# Set up dummy compiler flags targets
# ##############################################################################
include(HPXLocal_CompilerFlagsTargets)

# ##############################################################################
# Setup platform for which HPX should be compiled for.
#
include(HPXLocal_SetPlatform)
if("${HPXLocal_PLATFORM_UC}" STREQUAL "ANDROID")
  unset(HPXLocal_LIBRARY_VERSION)
  unset(HPXLocal_SOVERSION)
endif()

# Adjust a couple of build-system settings, if HPX is to be built using vcpkg
if(MSVC)
  set(_with_vcpkg_default OFF)
  if(VCPKG_TOOLCHAIN)
    set(_with_vcpkg_default ON)
  endif()
  hpx_local_option(
    HPXLocal_WITH_VCPKG
    BOOL
    "Build HPX in the context of the vcpkg build and configuration tool (default: OFF)."
    ${_with_vcpkg_default}
    ADVANCED
  )
  if(HPXLocal_WITH_VCPKG)
    hpx_local_add_config_define(HPX_HAVE_VCPKG)
  endif()

  hpx_local_option(
    HPXLocal_WITH_VS_STARTUP_PROJECT STRING
    "Define the startup project for the HPX solution (default: ALL_BUILD)."
    "ALL_BUILD" ADVANCED
  )
  if(HPXLocal_WITH_VS_STARTUP_PROJECT)
    set(VS_STARTUP_PROJECT ${HPXLocal_WITH_VS_STARTUP_PROJECT})
  endif()
endif()

# ##############################################################################

# ##############################################################################
# Set our build options cache variables which are customizable by users
#

hpx_local_option(
  HPXLocal_WITH_DEPRECATION_WARNINGS BOOL
  "Enable warnings for deprecated facilities. (default: ON)" ON ADVANCED
)

if(HPXLocal_WITH_DEPRECATION_WARNINGS)
  # enable deprecation warnings globally
  hpx_local_add_config_cond_define(HPX_LOCAL_HAVE_DEPRECATION_WARNINGS 1)
endif()

# Generic build options
set(DEFAULT_MALLOC "system")
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(DEFAULT_MALLOC "tcmalloc")
endif()

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(HPXLocal_WITH_STACKOVERFLOW_DETECTION_DEFAULT OFF)
  string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UC)
  if("${CMAKE_BUILD_TYPE_UC}" STREQUAL "DEBUG")
    set(HPXLocal_WITH_STACKOVERFLOW_DETECTION_DEFAULT ON)
  endif()
  hpx_local_option(
    HPXLocal_WITH_STACKOVERFLOW_DETECTION
    BOOL
    "Enable stackoverflow detection for HPX threads/coroutines. (default: OFF, debug: ON)"
    ${HPXLocal_WITH_STACKOVERFLOW_DETECTION_DEFAULT}
    ADVANCED
  )
  if(HPXLocal_WITH_STACKOVERFLOW_DETECTION)
    hpx_local_add_config_define(HPX_HAVE_STACKOVERFLOW_DETECTION)
  endif()
endif()

hpx_local_option(
  HPXLocal_WITH_MALLOC
  STRING
  "Define which allocator should be linked in. Options are: system, tcmalloc, jemalloc, mimalloc, tbbmalloc, and custom (default is: tcmalloc)"
  ${DEFAULT_MALLOC}
  STRINGS "system;tcmalloc;jemalloc;mimalloc;tbbmalloc;custom"
)

# On some systems jemalloc requires an explicit prefix for the API functions
# (i.e. 'malloc' is called 'je_malloc', etc.)
if(${HPXLocal_WITH_MALLOC} STREQUAL "jemalloc")
  if(MSVC)
    set(HPXLocal_WITH_JEMALLOC_PREFIX_DEFAULT "je_")
  else()
    set(HPXLocal_WITH_JEMALLOC_PREFIX_DEFAULT "<none>")
  endif()
  hpx_local_option(
    HPXLocal_WITH_JEMALLOC_PREFIX STRING
    "Optional naming prefix for jemalloc API functions"
    ${HPXLocal_WITH_JEMALLOC_PREFIX_DEFAULT} ADVANCED
  )
endif()

# Logging configuration
hpx_local_option(
  HPXLocal_WITH_LOGGING BOOL "Build HPX with logging enabled (default: ON)." ON
  ADVANCED
)
if(HPXLocal_WITH_LOGGING)
  hpx_local_add_config_define(HPX_HAVE_LOGGING)
endif()

# Compiler related build options
hpx_local_option(
  HPXLocal_WITH_GCC_VERSION_CHECK BOOL
  "Don't ignore version reported by gcc (default: ON)" ON ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_COMPILER_WARNINGS BOOL "Enable compiler warnings (default: ON)"
  ON ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_COMPILER_WARNINGS_AS_ERRORS BOOL
  "Turn compiler warnings into errors (default: OFF)" OFF ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_EXECUTABLE_PREFIX STRING
  "Executable prefix (default none), 'hpx_local_' useful for system install." ""
  CATEGORY "Build Targets"
)

if(WIN32)
  set(HPXLocal_WITH_PSEUDO_DEPENDENCIES
      OFF
      CACHE INTERNAL "" FORCE
  )
else()
  set(HPXLocal_WITH_PSEUDO_DEPENDENCIES
      ON
      CACHE INTERNAL "" FORCE
  )
endif()

hpx_local_option(
  HPXLocal_WITH_UNITY_BUILD BOOL
  "Enable unity build for certain build targets (default OFF)" OFF ADVANCED
)
if(HPXLocal_WITH_UNITY_BUILD)
  set(HPXLocal_WITH_UNITY_BUILD_OPTION UNITY_BUILD)
endif()

hpx_local_option(
  HPXLocal_WITH_PRECOMPILED_HEADERS
  BOOL
  "Enable precompiled headers for certain build targets (experimental) (default OFF)"
  OFF
  ADVANCED
)
if(HPXLocal_WITH_PRECOMPILED_HEADERS)
  set(HPXLocal_WITH_PRECOMPILED_HEADERS_INTERNAL ON)

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
      <boost/container/small_vector.hpp>
      <boost/fusion/include/vector.hpp>
      <boost/intrusive/slist.hpp>
      <boost/lockfree/queue.hpp>
      <boost/optional.hpp>
      <boost/spirit/home/x3.hpp>
      <boost/tokenizer.hpp>
      <boost/utility/string_ref.hpp>
  )

  if(HPXLocal_WITH_CXX17_FILESYSTEM)
    list(APPEND system_precompiled_headers <filesystem>)
  endif()

  set(system_precompiled_headers_dependencies
      hpx_local_dependencies_boost hpx_local_private_flags
      hpx_local_public_flags Asio::asio
  )

  add_library(hpx_local_precompiled_headers OBJECT libs/src/dummy.cpp)
  target_link_libraries(
    hpx_local_precompiled_headers
    PRIVATE hpx_local_public_flags hpx_local_private_flags
            hpx_local_base_libraries ${system_precompiled_headers_dependencies}
  )
  target_precompile_headers(
    hpx_local_precompiled_headers PRIVATE ${system_precompiled_headers}
  )

  set_target_properties(hpx_local_precompiled_headers PROPERTIES FOLDER "Core")
endif()

# ##############################################################################
hpx_local_option(
  HPXLocal_WITH_EXAMPLES BOOL "Build the HPX examples (default OFF)" OFF
  CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS BOOL "Build the HPX tests (default OFF)" OFF
  CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_BENCHMARKS BOOL "Build HPX benchmark tests (default ON)"
  ON CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_REGRESSIONS BOOL "Build HPX regression tests (default ON)"
  ON CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_UNIT BOOL "Build HPX unit tests (default ON)" ON
  CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_HEADERS BOOL "Build HPX header tests (default: ON)" ON
  ADVANCED CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_EXTERNAL_BUILD BOOL
  "Build external cmake build tests (default: ON)" ON ADVANCED
  CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_TESTS_EXAMPLES BOOL "Add HPX examples as tests (default: OFF)"
  OFF ADVANCED CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_COMPILE_ONLY_TESTS BOOL
  "Create build system support for compile time only HPX tests (default ON)" ON
  CATEGORY "Build Targets"
)
hpx_local_option(
  HPXLocal_WITH_FAIL_COMPILE_TESTS BOOL
  "Create build system support for fail compile HPX tests (default ON)" ON
  CATEGORY "Build Targets"
)

# disable all tests if HPXLocal_WITH_TESTS=OFF
if(NOT HPXLocal_WITH_TESTS)
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_BENCHMARKS
    VALUE OFF
    FORCE
  )
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_REGRESSIONS
    VALUE OFF
    FORCE
  )
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_UNIT
    VALUE OFF
    FORCE
  )
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_HEADERS
    VALUE OFF
    FORCE
  )
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_EXTERNAL_BUILD
    VALUE OFF
    FORCE
  )
  hpx_local_set_option(
    HPXLocal_WITH_TESTS_EXAMPLES
    VALUE OFF
    FORCE
  )
endif()

hpx_local_option(
  HPXLocal_WITH_TOOLS BOOL "Build HPX tools (default: OFF)" OFF ADVANCED
  CATEGORY "Build Targets"
)

set(HPXLocal_FULL_RPATH_DEFAULT ON)
if(APPLE OR WIN32)
  set(HPXLocal_FULL_RPATH_DEFAULT OFF)
endif()
hpx_local_option(
  HPXLocal_WITH_FULL_RPATH
  BOOL
  "Build and link HPX libraries and executables with full RPATHs (default: ${HPXLocal_FULL_RPATH_DEFAULT})"
  ${HPXLocal_FULL_RPATH_DEFAULT}
  ADVANCED
)

# ##############################################################################
# HPX CUDA Compute configuration
# ##############################################################################
# To avoid writing these strings out in multiple places, use vars to hold them
set(CUDA_OPTION_STRING "Enable support for CUDA (default: OFF)")
set(HIP_OPTION_STRING "Enable support for HIP (default: OFF)")
hpx_local_option(HPXLocal_WITH_CUDA BOOL "${CUDA_OPTION_STRING}" OFF ADVANCED)

# No need for the user to specify the option explicitly
hpx_local_option(HPXLocal_WITH_HIP BOOL "${HIP_OPTION_STRING}" OFF ADVANCED)
if("${CMAKE_CXX_COMPILER}" MATCHES "hipcc$")
  set(HPXLocal_WITH_HIP
      ON
      CACHE BOOL "${HIP_OPTION_STRING}" FORCE
  )
endif()

if(HPXLocal_WITH_CUDA AND HPXLocal_WITH_HIP)
  hpx_local_error(
    "HPXLocal_WITH_CUDA=ON and HPXLocal_WITH_HIP=ON. Only one of them can be on at the same time.\
 Note: HPXLocal_WITH_HIP is automatically enabled when compiling with hipcc."
  )
endif()

if(HPXLocal_WITH_CUDA OR HPXLocal_WITH_HIP)
  hpx_local_add_config_define(HPX_HAVE_GPU_SUPPORT)
  set(HPXLocal_WITH_GPU_SUPPORT ON)
  set(HPXLocal_WITH_GPU_SUPPORT
      ON
      CACHE INTERNAL "" FORCE
  )
else()
  set(HPXLocal_WITH_GPU_SUPPORT
      OFF
      CACHE INTERNAL "" FORCE
  )
endif()

# ##############################################################################
# HPX datapar configuration
# ##############################################################################
hpx_local_option(
  HPXLocal_WITH_DATAPAR_VC
  BOOL
  "Enable data parallel algorithm support using the external Vc library (default: OFF)"
  OFF
  ADVANCED
)
if(HPXLocal_WITH_DATAPAR_VC)
  hpx_local_option(
    HPXLocal_WITH_DATAPAR_VC_NO_LIBRARY BOOL
    "Don't link with the Vc static library (default: OFF)" OFF ADVANCED
  )
endif()

if(HPXLocal_WITH_DATAPAR_VC)
  hpx_local_warn(
    "Vc support is deprecated. This option will be removed in a future release. It will be replaced with SIMD support from the C++ standard library"
  )
  include(HPXLocal_SetupVc)
endif()
if(NOT HPXLocal_WITH_DATAPAR_VC)
  hpx_local_info("No vectorization library configured")
else()
  hpx_local_option(
    HPXLocal_WITH_DATAPAR BOOL
    "Enable data parallel algorithm support (default: ON)" ON ADVANCED
  )
endif()

# ##############################################################################
# Threadlevel Nice option
# ##############################################################################
hpx_local_option(
  HPXLocal_WITH_NICE_THREADLEVEL
  BOOL
  "Set HPX worker threads to have high NICE level (may impact performance) (default: OFF)"
  OFF
  ADVANCED
)
if(HPXLocal_WITH_NICE_THREADLEVEL)
  hpx_local_info("Nice threadlevel is enabled.")
  hpx_local_add_config_define(HPX_HAVE_NICE_THREADLEVEL)
else()
  hpx_local_info("Nice threadlevel is disabled.")
endif()

# ##############################################################################
# Utility configuration
# ##############################################################################
set(HPXLocal_HIDDEN_VISIBILITY_DEFAULT ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  if("${HPXLocal_PLATFORM_UC}" STREQUAL "ANDROID")
    set(HPXLocal_HIDDEN_VISIBILITY_DEFAULT OFF)
  endif()
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(HPXLocal_HIDDEN_VISIBILITY_DEFAULT OFF)
endif()
if(APPLE)
  set(HPXLocal_HIDDEN_VISIBILITY_DEFAULT OFF)
endif()

hpx_local_option(
  HPXLocal_WITH_HIDDEN_VISIBILITY
  BOOL
  "Use -fvisibility=hidden for builds on platforms which support it (default ${HPXLocal_HIDDEN_VISIBILITY_DEFAULT})"
  ${HPXLocal_HIDDEN_VISIBILITY_DEFAULT}
  ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_AUTOMATIC_SERIALIZATION_REGISTRATION
  BOOL
  "Use automatic serialization registration for actions and functions. This affects compatibility between HPX applications compiled with different compilers (default ON)"
  ON
  ADVANCED
)
if(HPXLocal_WITH_AUTOMATIC_SERIALIZATION_REGISTRATION)
  hpx_local_add_config_define(HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION)
endif()

hpx_local_option(
  HPXLocal_WITH_ZERO_COPY_SERIALIZATION_THRESHOLD
  STRING
  "The threshold in bytes to when perform zero copy optimizations (default: 128)"
  "128"
  ADVANCED
)
hpx_local_add_config_define(
  HPX_ZERO_COPY_SERIALIZATION_THRESHOLD
  ${HPXLocal_WITH_ZERO_COPY_SERIALIZATION_THRESHOLD}
)

hpx_local_option(
  HPXLocal_WITH_DISABLED_SIGNAL_EXCEPTION_HANDLERS
  BOOL
  "Disables the mechanism that produces debug output for caught signals and unhandled exceptions (default: OFF)"
  OFF
  ADVANCED
)
if(HPXLocal_WITH_DISABLED_SIGNAL_EXCEPTION_HANDLERS)
  hpx_local_add_config_define(HPX_HAVE_DISABLED_SIGNAL_EXCEPTION_HANDLERS)
endif()

# Thread Manager related build options

set(HPX_MAX_CPU_COUNT_DEFAULT "64")
hpx_local_option(
  HPXLocal_WITH_MAX_CPU_COUNT
  STRING
  "HPX applications will not use more that this number of OS-Threads (empty string means dynamic) (default: ${HPX_MAX_CPU_COUNT_DEFAULT})"
  "${HPX_MAX_CPU_COUNT_DEFAULT}"
  CATEGORY "Thread Manager"
  ADVANCED
)
if(HPXLocal_WITH_MAX_CPU_COUNT)
  hpx_local_add_config_define(
    HPX_HAVE_MAX_CPU_COUNT ${HPXLocal_WITH_MAX_CPU_COUNT}
  )
endif()
if((NOT HPXLocal_WITH_MAX_CPU_COUNT) OR (HPXLocal_WITH_MAX_CPU_COUNT GREATER 64
                                        )
)
  hpx_local_add_config_define(HPX_HAVE_MORE_THAN_64_THREADS)
endif()

set(HPX_MAX_NUMA_DOMAIN_COUNT_DEFAULT "8")
hpx_local_option(
  HPXLocal_WITH_MAX_NUMA_DOMAIN_COUNT
  STRING
  "HPX applications will not run on machines with more NUMA domains (default: ${HPX_MAX_NUMA_DOMAIN_COUNT_DEFAULT})"
  ${HPX_MAX_NUMA_DOMAIN_COUNT_DEFAULT}
  CATEGORY "Thread Manager"
  ADVANCED
)
hpx_local_add_config_define(
  HPX_HAVE_MAX_NUMA_DOMAIN_COUNT ${HPXLocal_WITH_MAX_NUMA_DOMAIN_COUNT}
)

hpx_local_option(
  HPXLocal_WITH_THREAD_STACK_MMAP BOOL
  "Use mmap for stack allocation on appropriate platforms" ON
  CATEGORY "Thread Manager"
  ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_THREAD_MANAGER_IDLE_BACKOFF BOOL
  "HPX scheduler threads do exponential backoff on idle queues (default: ON)" ON
  CATEGORY "Thread Manager"
  ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_STACKTRACES BOOL
  "Attach backtraces to HPX exceptions (default: ON)" ON
  CATEGORY "Thread Manager"
  ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_THREAD_BACKTRACE_ON_SUSPENSION BOOL
  "Enable thread stack back trace being captured on suspension (default: OFF)"
  OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

# We create a target to contain libraries like rt, dl etc. in order to remove
# global variables
add_library(hpx_local_base_libraries INTERFACE)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  target_link_libraries(hpx_local_base_libraries INTERFACE imf svml irng intlc)
endif()

if(HPXLocal_WITH_STACKTRACES OR HPXLocal_WITH_THREAD_BACKTRACE_ON_SUSPENSION)
  hpx_local_info("Stack traces are enabled.")
  hpx_local_add_config_define(HPX_HAVE_STACKTRACES)
  if(WIN32)
    target_link_libraries(hpx_local_base_libraries INTERFACE dbghelp)
  endif()

  hpx_local_option(
    HPXLocal_WITH_THREAD_BACKTRACE_DEPTH STRING
    "Thread stack back trace depth being captured (default: 20)" "20"
    CATEGORY "Thread Manager"
    ADVANCED
  )
  hpx_local_add_config_define(
    HPX_HAVE_THREAD_BACKTRACE_DEPTH ${HPXLocal_WITH_THREAD_BACKTRACE_DEPTH}
  )

  if(("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
     AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
          OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  )

    # This option is OFF by default as we have seen random segfaults out in the
    # wild if this is enabled.
    hpx_local_option(
      HPXLocal_WITH_STACKTRACES_STATIC_SYMBOLS BOOL
      "Thread stack back trace will resolve static symbols (default: OFF)" OFF
      CATEGORY "Thread Manager"
      ADVANCED
    )
    hpx_local_add_config_define(
      HPX_HAVE_STACKTRACES_STATIC_SYMBOLS
      ${HPXLocal_WITH_STACKTRACES_STATIC_SYMBOLS}
    )

    # Demangling can segfault in certain configurations.
    hpx_local_option(
      HPXLocal_WITH_STACKTRACES_DEMANGLE_SYMBOLS BOOL
      "Thread stack back trace symbols will be demangled (default: ON)" ON
      CATEGORY "Thread Manager"
      ADVANCED
    )
    if(HPXLocal_WITH_STACKTRACES_DEMANGLE_SYMBOLS)
      hpx_local_add_config_define(HPX_HAVE_STACKTRACES_DEMANGLE_SYMBOLS)
    endif()
  endif()
endif()

if(HPXLocal_WITH_THREAD_BACKTRACE_ON_SUSPENSION)
  hpx_local_add_config_define(HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION)

  hpx_local_option(
    HPXLocal_WITH_THREAD_FULLBACKTRACE_ON_SUSPENSION BOOL
    "Enable thread stack back trace being captured on suspension (default: OFF)"
    OFF
    CATEGORY "Thread Manager"
    ADVANCED
  )
  if(HPXLocal_WITH_THREAD_FULLBACKTRACE_ON_SUSPENSION)
    hpx_local_add_config_define(HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION)
  endif()
endif()

hpx_local_option(
  HPXLocal_WITH_THREAD_TARGET_ADDRESS BOOL
  "Enable storing target address in thread for NUMA awareness (default: OFF)"
  OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(HPXLocal_WITH_THREAD_TARGET_ADDRESS)
  hpx_local_add_config_define(HPX_HAVE_THREAD_TARGET_ADDRESS)
endif()

hpx_local_option(
  HPXLocal_WITH_THREAD_QUEUE_WAITTIME BOOL
  "Enable collecting queue wait times for threads (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(HPXLocal_WITH_THREAD_QUEUE_WAITTIME)
  hpx_local_add_config_define(HPX_HAVE_THREAD_QUEUE_WAITTIME)
endif()

hpx_local_option(
  HPXLocal_WITH_THREAD_IDLE_RATES
  BOOL
  "Enable measuring the percentage of overhead times spent in the scheduler (default: OFF)"
  OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(HPXLocal_WITH_THREAD_IDLE_RATES)
  hpx_local_option(
    HPXLocal_WITH_BACKGROUND_THREAD_COUNTERS
    BOOL
    "Enable performance counters related to adaptive parcel coalescing (default: OFF)."
    OFF
    ADVANCED
  )
  if(HPXLocal_WITH_BACKGROUND_THREAD_COUNTERS)
    hpx_local_add_config_define(HPX_HAVE_BACKGROUND_THREAD_COUNTERS)
  endif()
endif()

hpx_local_option(
  HPXLocal_WITH_THREAD_CREATION_AND_CLEANUP_RATES BOOL
  "Enable measuring thread creation and cleanup times (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(HPXLocal_WITH_THREAD_IDLE_RATES)
  hpx_local_add_config_define(HPX_HAVE_THREAD_IDLE_RATES)
  if(HPXLocal_WITH_THREAD_CREATION_AND_CLEANUP_RATES)
    hpx_local_add_config_define(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
  endif()
endif()

hpx_local_option(
  HPXLocal_WITH_THREAD_CUMULATIVE_COUNTS
  BOOL
  "Enable keeping track of cumulative thread counts in the schedulers (default: ON)"
  ON
  CATEGORY "Thread Manager"
  ADVANCED
)

if(HPXLocal_WITH_THREAD_CUMULATIVE_COUNTS)
  hpx_local_add_config_define(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
endif()

hpx_local_option(
  HPXLocal_WITH_THREAD_STEALING_COUNTS
  BOOL
  "Enable keeping track of counts of thread stealing incidents in the schedulers (default: OFF)"
  OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(HPXLocal_WITH_THREAD_STEALING_COUNTS)
  hpx_local_add_config_define(HPX_HAVE_THREAD_STEALING_COUNTS)
endif()

hpx_local_option(
  HPXLocal_WITH_COROUTINE_COUNTERS BOOL
  "Enable keeping track of coroutine creation and rebind counts (default: OFF)"
  OFF
  CATEGORY "Thread Manager"
  ADVANCED
)
if(HPXLocal_WITH_COROUTINE_COUNTERS)
  hpx_local_add_config_define(HPX_HAVE_COROUTINE_COUNTERS)
endif()

hpx_local_option(
  HPXLocal_WITH_THREAD_LOCAL_STORAGE BOOL
  "Enable thread local storage for all HPX threads (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(HPXLocal_WITH_THREAD_LOCAL_STORAGE)
  hpx_local_add_config_define(HPX_HAVE_THREAD_LOCAL_STORAGE)
endif()

hpx_local_option(
  HPXLocal_WITH_SCHEDULER_LOCAL_STORAGE BOOL
  "Enable scheduler local storage for all HPX schedulers (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(HPXLocal_WITH_SCHEDULER_LOCAL_STORAGE)
  hpx_local_add_config_define(HPX_HAVE_SCHEDULER_LOCAL_STORAGE)
endif()

hpx_local_option(
  HPXLocal_WITH_SPINLOCK_POOL_NUM STRING
  "Number of elements a spinlock pool manages (default: 128)" 128
  CATEGORY "Thread Manager"
  ADVANCED
)

hpx_local_add_config_define(
  HPX_HAVE_SPINLOCK_POOL_NUM ${HPXLocal_WITH_SPINLOCK_POOL_NUM}
)

hpx_local_option(
  HPXLocal_WITH_SPINLOCK_DEADLOCK_DETECTION BOOL
  "Enable spinlock deadlock detection (default: OFF)" OFF
  CATEGORY "Thread Manager"
  ADVANCED
)

if(HPXLocal_WITH_SPINLOCK_DEADLOCK_DETECTION)
  hpx_local_add_config_define(HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION)
endif()

# Options for automatically fetching Asio
hpx_local_option(
  HPXLocal_WITH_FETCH_ASIO
  BOOL
  "Use FetchContent to fetch Asio. By default an installed Asio will be used. (default: OFF)"
  OFF
  CATEGORY "Build Targets"
  ADVANCED
)
hpx_local_option(
  HPXLocal_WITH_ASIO_TAG STRING "Asio repository tag or branch" "asio-1-18-2"
  CATEGORY "Build Targets"
  ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_PAPI BOOL "Enable PAPI support." OFF CATEGORY "Profiling"
)
if(HPXLocal_WITH_PAPI)
  if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    hpx_local_error(
      "HPXLocal_WITH_PAPI was set to ON, but PAPI can only be used on Linux (this is ${CMAKE_SYSTEM_NAME})"
    )
  endif()
  hpx_local_add_config_define(HPX_HAVE_PAPI)
endif()

hpx_local_option(
  HPXLocal_WITH_ITTNOTIFY BOOL "Enable Amplifier (ITT) instrumentation support."
  OFF CATEGORY "Profiling"
)

# Experimental settings
hpx_local_option(
  HPXLocal_WITH_IO_POOL
  BOOL
  "Disable internal IO thread pool, do not change if not absolutely necessary (default: ON)"
  ON
  CATEGORY "Thread Manager"
  ADVANCED
)
if(HPXLocal_WITH_IO_POOL)
  hpx_local_add_config_define(HPX_HAVE_IO_POOL)
endif()

hpx_local_option(
  HPXLocal_WITH_TIMER_POOL
  BOOL
  "Disable internal timer thread pool, do not change if not absolutely necessary (default: ON)"
  ON
  CATEGORY "Thread Manager"
  ADVANCED
)
if(HPXLocal_WITH_TIMER_POOL)
  hpx_local_add_config_define(HPX_HAVE_TIMER_POOL)
endif()

hpx_local_option(
  HPXLocal_WITH_ASYNC_MPI
  BOOL
  "Enable support for returning futures from MPI asynchronous calls (default: ON)"
  OFF
  CATEGORY "MPI"
)

if(HPXLocal_WITH_ASYNC_MPI)
  # mpi parcelport settings
  hpx_local_option(
    HPXLocal_WITH_PARCELPORT_MPI_ENV
    STRING
    "List of environment variables checked to detect MPI (default: MV2_COMM_WORLD_RANK;PMI_RANK;OMPI_COMM_WORLD_SIZE;ALPS_APP_PE;PMIX_RANK)."
    "MV2_COMM_WORLD_RANK;PMI_RANK;OMPI_COMM_WORLD_SIZE;ALPS_APP_PE;PMIX_RANK"
    CATEGORY "Parcelport"
    ADVANCED
  )

  # This list is to detect whether we run inside an mpi environment. If one of
  # those environment variables is set, the MPI parcelport is enabled by
  # default. PMI_RANK: Intel MPI and MVAPICH2 OMPI_COMM_WORLD_SIZE: OpenMPI
  # starting at version 1.3
  if(HPXLocal_WITH_PARCELPORT_MPI_ENV)
    string(REPLACE ";" "," hpx_local_parcelport_mpi_env_
                   "${HPXLocal_WITH_PARCELPORT_MPI_ENV}"
    )
    hpx_local_add_config_define(
      HPX_HAVE_PARCELPORT_MPI_ENV "\"${hpx_local_parcelport_mpi_env_}\""
    )
  endif()

  hpx_local_option(
    HPXLocal_WITH_PARCELPORT_MPI_MULTITHREADED BOOL
    "Turn on MPI multithreading support (default: ON)." ON
    CATEGORY "Parcelport"
    ADVANCED
  )
  if(HPXLocal_WITH_PARCELPORT_MPI_MULTITHREADED)
    hpx_local_add_config_define(HPX_HAVE_PARCELPORT_MPI_MULTITHREADED)
  endif()

  if(MSVC)
    # FIXME: add OpenMPI specific flag here for now as the
    # hpx_local_add_compile_flag() below does not add the extra options to the
    # top level directory
    hpx_local_add_config_define(OMPI_IMPORTS)
  endif()
endif()

# External libraries/frameworks used by sme of the examples and benchmarks
hpx_local_option(
  HPXLocal_WITH_EXAMPLES_OPENMP BOOL
  "Enable examples requiring OpenMP support (default: OFF)." OFF
  CATEGORY "Build Targets"
  ADVANCED
)
if(HPXLocal_WITH_EXAMPLES_OPENMP)
  find_package(OpenMP REQUIRED)
endif()
hpx_local_option(
  HPXLocal_WITH_EXAMPLES_TBB BOOL
  "Enable examples requiring TBB support (default: OFF)." OFF
  CATEGORY "Build Targets"
  ADVANCED
)
if(HPXLocal_WITH_EXAMPLES_TBB)
  find_package(TBB)
  if(NOT TBB_FOUND)
    set(HPXLocal_WITH_EXAMPLES_TBB OFF)
  endif()
endif()
hpx_local_option(
  HPXLocal_WITH_EXAMPLES_QTHREADS BOOL
  "Enable examples requiring QThreads support (default: OFF)." OFF
  CATEGORY "Build Targets"
  ADVANCED
)
if(HPXLocal_WITH_EXAMPLES_QTHREADS)
  find_package(QThreads)
  if(NOT QTHREADS_FOUND)
    set(HPXLocal_WITH_EXAMPLES_QTHREADS OFF)
  endif()
endif()
hpx_local_option(
  HPXLocal_WITH_EXAMPLES_HDF5 BOOL
  "Enable examples requiring HDF5 support (default: OFF)." OFF
  CATEGORY "Build Targets"
  ADVANCED
)
if(HPXLocal_WITH_EXAMPLES_HDF5)
  find_package(HDF5 COMPONENTS CXX)
  if(NOT HDF5_FOUND)
    set(HPXLocal_WITH_EXAMPLES_HDF5 OFF)
  endif()
endif()

# Disabling the Qt example on BG/Q as GUIs don't make sense there anyways
if(NOT "${HPXLocal_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
  hpx_local_option(
    HPXLocal_WITH_EXAMPLES_QT4 BOOL
    "Enable examples requiring Qt4 support (default: OFF)." OFF
    CATEGORY "Build Targets"
    ADVANCED
  )
  if(HPXLocal_WITH_EXAMPLES_QT4)
    find_package(Qt4)
    if(NOT QT4_FOUND)
      set(HPXLocal_WITH_EXAMPLES_QT4 OFF)
    endif()
  endif()
endif()

# Debugging related build options
hpx_local_option(
  HPXLocal_WITH_VALGRIND BOOL "Enable Valgrind instrumentation support." OFF
  CATEGORY "Debugging"
)

hpx_local_option(
  HPXLocal_WITH_SANITIZERS BOOL
  "Configure with sanitizer instrumentation support." OFF CATEGORY "Debugging"
)

hpx_local_option(
  HPXLocal_WITH_VERIFY_LOCKS
  BOOL
  "Enable lock verification code (default: OFF, implicitly enabled in debug builds)"
  OFF
  CATEGORY "Debugging"
  ADVANCED
)
hpx_local_option(
  HPXLocal_WITH_VERIFY_LOCKS_GLOBALLY
  BOOL
  "Enable global lock verification code (default: OFF, implicitly enabled in debug builds)"
  OFF
  CATEGORY "Debugging"
  ADVANCED
)
hpx_local_option(
  HPXLocal_WITH_VERIFY_LOCKS_BACKTRACE
  BOOL
  "Enable thread stack back trace being captured on lock registration (to be used in combination with HPXLocal_WITH_VERIFY_LOCKS=ON, default: OFF)"
  OFF
  CATEGORY "Debugging"
  ADVANCED
)
hpx_local_option(
  HPXLocal_WITH_THREAD_DEBUG_INFO
  BOOL
  "Enable thread debugging information (default: OFF, implicitly enabled in debug builds)"
  OFF
  CATEGORY "Debugging"
  ADVANCED
)
hpx_local_option(
  HPXLocal_WITH_THREAD_GUARD_PAGE BOOL "Enable thread guard page (default: ON)"
  ON
  CATEGORY "Debugging"
  ADVANCED
)

if(HPXLocal_WITH_VERIFY_LOCKS)
  hpx_local_add_config_define(HPX_HAVE_VERIFY_LOCKS)
  if(HPXLocal_WITH_VERIFY_LOCKS_BACKTRACE)
    hpx_local_add_config_define(HPX_HAVE_VERIFY_LOCKS_BACKTRACE)
  endif()
endif()
if(HPXLocal_WITH_VERIFY_LOCKS_GLOBALLY)
  hpx_local_add_config_define(HPX_HAVE_VERIFY_LOCKS_GLOBALLY)
endif()

# Additional debug support
if(NOT WIN32 AND HPXLocal_WITH_THREAD_GUARD_PAGE)
  hpx_local_add_config_define(HPX_HAVE_THREAD_GUARD_PAGE)
endif()

if(NOT WIN32 AND HPXLocal_WITH_THREAD_STACK_MMAP)
  hpx_local_add_config_define(HPX_HAVE_THREAD_STACK_MMAP)
endif()

if(HPXLocal_WITH_THREAD_MANAGER_IDLE_BACKOFF)
  hpx_local_add_config_define(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
endif()

hpx_local_option(
  HPXLocal_WITH_THREAD_DESCRIPTION_FULL BOOL
  "Use function address for thread description (default: OFF)" OFF
  CATEGORY "Debugging"
  ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_ATTACH_DEBUGGER_ON_TEST_FAILURE BOOL
  "Break the debugger if a test has failed  (default: OFF)" OFF
  CATEGORY "Debugging"
  ADVANCED
)
if(HPXLocal_WITH_ATTACH_DEBUGGER_ON_TEST_FAILURE)
  hpx_local_add_config_define(HPX_HAVE_ATTACH_DEBUGGER_ON_TEST_FAILURE)
endif()

hpx_local_option(
  HPXLocal_WITH_TESTS_DEBUG_LOG BOOL
  "Turn on debug logs (--hpx:debug-hpx-log) for tests (default: OFF)" OFF
  CATEGORY "Debugging"
  ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_TESTS_DEBUG_LOG_DESTINATION STRING
  "Destination for test debug logs (default: cout)" "cout"
  CATEGORY "Debugging"
  ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_TESTS_MAX_THREADS_PER_LOCALITY
  STRING
  "Maximum number of threads to use for tests (default: 0, use the number of threads specified by the test)"
  0
  CATEGORY "Debugging"
  ADVANCED
)

hpx_local_option(
  HPXLocal_WITH_PARALLEL_TESTS_BIND_NONE
  BOOL
  "Pass --hpx:bind=none to tests that may run in parallel (cmake -j flag) (default: OFF)"
  OFF
  CATEGORY "Debugging"
  ADVANCED
)

# If APEX is defined, the action timers need thread debug info.
hpx_local_option(
  HPXLocal_WITH_APEX BOOL "Enable APEX instrumentation support." OFF
  CATEGORY "Profiling"
)
if(HPXLocal_WITH_APEX)
  hpx_local_add_config_define(HPX_HAVE_APEX) # tell HPX that we use APEX
  hpx_local_add_config_define(HPX_HAVE_THREAD_DESCRIPTION)
  hpx_local_add_config_define(HPX_HAVE_THREAD_PARENT_REFERENCE)
  if(HPXLocal_WITH_THREAD_DESCRIPTION_FULL)
    hpx_local_add_config_define(HPX_HAVE_THREAD_DESCRIPTION_FULL)
  endif()
endif()

if(HPXLocal_WITH_THREAD_DEBUG_INFO)
  hpx_local_add_config_define(HPX_HAVE_THREAD_PARENT_REFERENCE)
  hpx_local_add_config_define(HPX_HAVE_THREAD_PHASE_INFORMATION)
  hpx_local_add_config_define(HPX_HAVE_THREAD_DESCRIPTION)
  hpx_local_add_config_define(HPX_HAVE_THREAD_DEADLOCK_DETECTION)
  if(HPXLocal_WITH_THREAD_DESCRIPTION_FULL)
    hpx_local_add_config_define(HPX_HAVE_THREAD_DESCRIPTION_FULL)
  endif()
endif()

# Developer tools
hpx_local_option(
  HPXLocal_WITH_VIM_YCM BOOL
  "Generate HPX completion file for VIM YouCompleteMe plugin" OFF ADVANCED
)

# ##############################################################################
# Backwards compatibility options (edit for each release)

# BADBAD: This enables an overload of swap which is necessary to work around the
# problems caused by zip_iterator not being a real random access iterator.
# Dereferencing zip_iterator does not yield a true reference but only a
# temporary tuple holding true references.
#
# A real fix for this problem is proposed in PR0022R0
# (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0022r0.html)
#
hpx_local_option(
  HPXLocal_WITH_TUPLE_RVALUE_SWAP
  BOOL
  "Enable swapping of rvalue tuples (needed for parallel::sort_by_key, default: ON)."
  ON
  CATEGORY "Utility"
  ADVANCED
)
if(HPXLocal_WITH_TUPLE_RVALUE_SWAP)
  hpx_local_add_config_define(HPX_HAVE_TUPLE_RVALUE_SWAP)
endif()

# ##############################################################################
# Special diagnostic flags, do not enable by default, only if needed
hpx_local_option(
  HPXLocal_WITH_CHECK_MODULE_DEPENDENCIES
  BOOL
  "Verify that no modules are cross-referenced from a different module category \
  (default: OFF)"
  OFF
  ADVANCED
)

# Check for compiler compatibility
#

# Check if the selected compiler versions are supposed to work with our codebase
if(CMAKE_COMPILER_IS_GNUCXX AND HPXLocal_WITH_GCC_VERSION_CHECK)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    hpx_local_error(
      "GCC 7.0 or higher is required. Specify HPXLocal_WITH_GCC_VERSION_CHECK=OFF to ignore this error."
    )
  endif()
endif()

if(MSVC)
  if(MSVC_VERSION LESS 1900)
    hpx_local_error("MSVC x64 2015 or higher is required.")
  elseif(NOT CMAKE_CL_64)
    hpx_local_warn(
      "MSVC (32Bit) will compile but will fail running larger applications because of limitations in the Windows OS."
    )
  endif()
endif()

# Setup platform specific compiler options and check for compatible compilers
if("${HPXLocal_PLATFORM_UC}" STREQUAL "NATIVE")
  hpx_local_info("Compiling with the native toolset")
elseif("${HPXLocal_PLATFORM_UC}" STREQUAL "ANDROID")
  hpx_local_info("Compiling for Android devices")
elseif("${HPXLocal_PLATFORM_UC}" STREQUAL "XEONPHI")
  hpx_local_info("Compiling for Intel Xeon Phi devices")
  if(NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
    hpx_local_error(
      "HPX on the MIC can only be compiled with the Intel compiler."
    )
  endif()
elseif("${HPXLocal_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    hpx_local_error("HPX on the BG/Q can only be compiled with bgclang")
  endif()
  hpx_local_info("Compiling for BlueGene/Q")
endif()

if((NOT HPXLocal_WITH_CXX20_COROUTINES)
   AND ((MSVC_VERSION GREATER_EQUAL 1900) AND CMAKE_CL_64)
   OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
)

  # If C++20 coroutines are not supported by the compiler there is still a
  # chance to enable them through command line options.

  hpx_local_option(
    HPXLocal_WITH_AWAIT BOOL
    "Enable the use of experimental co_await functionality" OFF ADVANCED
    CATEGORY "LCOs"
  )
  if(HPXLocal_WITH_AWAIT)
    hpx_local_add_config_define(HPX_HAVE_AWAIT)

    if((MSVC_VERSION GREATER_EQUAL 1900) AND CMAKE_CL_64)
      hpx_local_add_target_compile_option(-await PUBLIC)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      hpx_local_add_target_compile_option(-Xclang -fcoroutines-ts PUBLIC)

      # older Clang versions don't have <experimental/coroutine> yet
      hpx_local_option(
        HPXLocal_WITH_EMULATE_COROUTINE_SUPPORT_LIBRARY BOOL
        "Use hpx/util/await_traits.hpp instead of <experimental/coroutine>" OFF
        ADVANCED CATEGORY "LCOs"
      )
      if(HPXLocal_WITH_EMULATE_COROUTINE_SUPPORT_LIBRARY)
        hpx_local_add_config_define(HPX_HAVE_EMULATE_COROUTINE_SUPPORT_LIBRARY)
      endif()
    endif()
  endif()
endif()

# ##############################################################################
# Some special handling of the compilation is need on build infrastructure for
# generating packages for target architecture, see issue #3575

hpx_local_option(
  HPXLocal_WITH_BUILD_BINARY_PACKAGE
  BOOL
  "Build HPX on the build infrastructure on any LINUX distribution (default: OFF)."
  OFF
  ADVANCED
)

# ##############################################################################
# C++ feature tests
# ##############################################################################
# Need to include the CUDA setup before the config test to enable the CUDA
# language
include(HPXLocal_SetupCUDA)
include(HPXLocal_PerformCxxFeatureTests)
hpx_local_perform_cxx_feature_tests()

# ##############################################################################
# Set configuration option to use Boost.Context or not. This depends on the
# platform.
set(__use_generic_coroutine_context OFF)
if(APPLE)
  set(__use_generic_coroutine_context ON)
endif()
if("${HPXLocal_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
  set(__use_generic_coroutine_context ON)
endif()
hpx_local_option(
  HPXLocal_WITH_GENERIC_CONTEXT_COROUTINES
  BOOL
  "Use Boost.Context as the underlying coroutines context switch implementation."
  ${__use_generic_coroutine_context}
  ADVANCED
)

# ##############################################################################
# check for miscellaneous things
# ##############################################################################

hpx_local_check_for_mm_prefetch(DEFINITIONS HPX_HAVE_MM_PREFETCH)

hpx_local_check_for_stable_inplace_merge(
  DEFINITIONS HPX_HAVE_STABLE_INPLACE_MERGE
)

if(NOT HPXLocal_WITH_STABLE_INPLACE_MERGE)
  hpx_local_warn(
    "The standard library you are using (libc++ version < 6) does not have a stable inplace_merge implementation."
  )
endif()

# ##############################################################################
# Check for misc system headers
# ##############################################################################

hpx_local_check_for_unistd_h(DEFINITIONS HPX_HAVE_UNISTD_H)

if(NOT WIN32)
  # ############################################################################
  # Macro definitions for system headers
  # ############################################################################
  add_definitions(-D_GNU_SOURCE)

  # ############################################################################
  # System libraries
  # ############################################################################
  if(NOT MSVC)
    if(HPXLocal_CXX11_STD_ATOMIC_LIBRARIES)
      target_link_libraries(
        hpx_local_base_libraries
        INTERFACE ${HPXLocal_CXX11_STD_ATOMIC_LIBRARIES}
      )
    endif()
  endif()

  if(HPX_HAVE_LIBSUPCPP)
    target_link_libraries(hpx_local_base_libraries INTERFACE supc++)
  endif()

  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
    target_link_libraries(hpx_local_base_libraries INTERFACE dl)
  endif()

  if(NOT APPLE AND NOT ("${HPXLocal_PLATFORM_UC}" STREQUAL "ANDROID"))
    target_link_libraries(hpx_local_base_libraries INTERFACE rt)
  endif()

  if("${HPXLocal_PLATFORM_UC}" STREQUAL "ANDROID")
    target_link_libraries(hpx_local_base_libraries INTERFACE log)
  endif()

  if(APPLE)
    hpx_local_add_compile_flag_if_available(-ftemplate-depth=256)
  endif()
endif()

if(WIN32)
  if(MSVC)
    hpx_local_add_target_compile_option(-Ox PUBLIC CONFIGURATIONS Release)

    # even VS2017 has an ICE when compiling with -Ob2
    hpx_local_add_target_compile_option(-Ob1 PUBLIC CONFIGURATIONS Release)

    # /RTC1 is incompatible with /await
    if((NOT HPXLocal_WITH_AWAIT) AND (NOT HPXLocal_WITH_CXX20_COROUTINES))
      hpx_local_add_target_compile_option(/RTC1 PUBLIC CONFIGURATIONS Debug)
    else()
      hpx_local_remove_target_compile_option(/RTC1 PUBLIC CONFIGURATIONS Debug)
    endif()

    # VS2012 and above has a special flag for improving the debug experience by
    # adding more symbol information to the build (-d2Zi)
    hpx_local_add_target_compile_option(
      -d2Zi+ PUBLIC CONFIGURATIONS RelWithDebInfo
    )

    # VS2013 and above know how to do link time constant data segment folding
    # VS2013 update 2 and above know how to remove debug information for
    # non-referenced functions and data (-Zc:inline)
    hpx_local_add_target_compile_option(-Zc:inline PUBLIC)
    hpx_local_add_target_compile_option(
      -Gw PUBLIC CONFIGURATIONS Release RelWithDebInfo MinSizeRel
    )
    hpx_local_add_target_compile_option(
      -Zo PUBLIC CONFIGURATIONS RelWithDebInfo
    )
    if(HPXLocal_WITH_DATAPAR_VC)
      hpx_local_add_target_compile_option(-std:c++latest PUBLIC)
      hpx_local_add_config_cond_define(_HAS_AUTO_PTR_ETC 1)
    endif()

    # Exceptions
    hpx_local_add_target_compile_option(-EHsc)
    if(MSVC_VERSION GREATER_EQUAL 1900)
      # assume conforming (throwing) operator new implementations
      hpx_local_add_target_compile_option(-Zc:throwingNew PUBLIC)

      # enable faster linking (requires VS2015 Update1) disabled for now as this
      # flag crashes debugger hpx_local_remove_link_flag(/debug CONFIGURATIONS
      # Debug) hpx_local_add_link_flag(/debug:fastlink CONFIGURATIONS Debug)

      # Update 2 requires to set _ENABLE_ATOMIC_ALIGNMENT_FIX for it to compile
      # atomics
      hpx_local_add_config_cond_define(_ENABLE_ATOMIC_ALIGNMENT_FIX)

      # Update 3 allows to flag rvalue misuses and enforces strict string const-
      # qualification conformance
      hpx_local_add_target_compile_option(-Zc:rvalueCast PUBLIC)
      hpx_local_add_target_compile_option(-Zc:strictStrings PUBLIC)

      # Update 8 requires to set _ENABLE_EXTENDED_ALIGNED_STORAGE for it to
      # compile atomics
      hpx_local_add_config_cond_define(_ENABLE_EXTENDED_ALIGNED_STORAGE)

      # Make sure that __cplusplus is properly defined
      hpx_local_add_target_compile_option(-Zc:__cplusplus PUBLIC)

      # Silence C++17 deprecation warnings
      hpx_local_add_config_cond_define(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

      # ASan is available in Visual Studion starting V16.8
      if((MSVC_VERSION GREATER_EQUAL 1928) AND HPXLocal_WITH_SANITIZERS)
        hpx_local_add_target_compile_option(
          -fsanitize=address PUBLIC CONFIGURATIONS RelWithDebInfo Debug
        )
      endif()
    endif()

    # Runtime type information
    hpx_local_add_target_compile_option(-GR PUBLIC)
    # Multiprocessor build
    hpx_local_add_target_compile_option(-MP PUBLIC)
    # Increase the maximum size of object file sections
    hpx_local_add_target_compile_option(-bigobj PUBLIC)
  endif()

  target_link_libraries(hpx_local_base_libraries INTERFACE psapi shlwapi)

  # ############################################################################
  # Macro definitions for system headers
  # ############################################################################
  add_definitions(-D_WINDOWS)
  add_definitions(-D_WIN32)
  hpx_local_add_config_cond_define(_WIN32_WINNT 0x0602)
  hpx_local_add_config_cond_define(_SCL_SECURE_NO_WARNINGS)
  hpx_local_add_config_cond_define(_CRT_SECURE_NO_WARNINGS)
  hpx_local_add_config_cond_define(_SCL_SECURE_NO_DEPRECATE)
  hpx_local_add_config_cond_define(_CRT_SECURE_NO_DEPRECATE)
  hpx_local_add_config_cond_define(_CRT_NONSTDC_NO_WARNINGS)
  hpx_local_add_config_cond_define(_WINSOCK_DEPRECATED_NO_WARNINGS)
  hpx_local_add_config_cond_define(_CRT_NON_CONFORMING_SWPRINTFS)
  hpx_local_add_config_cond_define(_SILENCE_FPOS_SEEKPOS_DEPRECATION_WARNING)

  # ############################################################################
  # Boost
  # ############################################################################

  hpx_local_add_config_cond_define(BOOST_USE_WINDOWS_H)
  if(NOT CMAKE_CL_64)
    hpx_local_add_config_cond_define(BOOST_NO_ALIGNMENT)
  endif()
  if(NOT HPXLocal_WITH_GENERIC_CONTEXT_COROUTINES)
    hpx_local_add_config_define(HPX_HAVE_FIBER_BASED_COROUTINES)
  endif()
  hpx_local_add_config_cond_define(PSAPI_VERSION 1)
endif()

# Configure Warnings
if(HPXLocal_WITH_COMPILER_WARNINGS)
  if(MSVC) # Adding special warning settings for the MSVC compiler ...
    hpx_local_add_compile_flag(-W3)

    # MSVC2012/2013 are overeager to report 'qualifier applied to function type
    # has no meaning; ignored'
    hpx_local_add_compile_flag(-wd4180)

    # Boost.Lockfree triggers 'warning C4307: '+' : integral constant overflow'
    # which is benign
    hpx_local_add_compile_flag(-wd4307)

    # object allocated on the heap may not be aligned
    hpx_local_add_compile_flag(-wd4316)

    # max symbol length exceeded
    hpx_local_add_compile_flag(-wd4503)

    # 'int': forcing value to bool 'true' or 'false' (performance warning)
    hpx_local_add_compile_flag(-wd4800)

    # vcpkg enables the /utf-8 option which causes (benign) warnings in the
    # Spirit headers
    if(HPXLocal_WITH_VCPKG)
      # The file contains a character starting at offset ... that is illegal in
      # the current source character set
      hpx_local_add_compile_flag(-wd4828)
    endif()

    if(HPXLocal_WITH_DATAPAR_VC)
      # unary minus operator applied to unsigned type, result still unsigned
      hpx_local_add_compile_flag(-wd4146)

      # '<=': signed/unsigned mismatch
      hpx_local_add_compile_flag(-wd4018)

      # 'return': conversion from 'short' to 'Vc_1::schar', possible loss of
      # data
      hpx_local_add_compile_flag(-wd4244)
    endif()

  else() # Trial and error approach for any other compiler ...
    hpx_local_add_compile_flag_if_available(-Wall)
    hpx_local_add_compile_flag_if_available(-Wextra)
    hpx_local_add_compile_flag_if_available(-Wno-strict-aliasing)
    hpx_local_add_compile_flag_if_available(-Wno-sign-promo)
    hpx_local_add_compile_flag_if_available(-Wno-attributes)
    hpx_local_add_compile_flag_if_available(-Wno-cast-align)

    # We do not in general guarantee ABI compatibility between C++ standards, so
    # we ignore this warning
    hpx_local_add_compile_flag_if_available(-Wno-noexcept-type)

    # Be extra strict about format checks Boost.Logging is built on fprintf,
    # sadly
    hpx_local_add_compile_flag_if_available(-Wformat=2)
    hpx_local_add_compile_flag_if_available(-Wno-format-nonliteral)

    # Self initialization is dangerous
    hpx_local_add_compile_flag_if_available(-Winit-self)

    # For portability
    hpx_local_add_compile_flag_if_available(-Wdouble-promotion)

    # Warn about casting that violates qualifiers or alignment
    hpx_local_add_compile_flag_if_available(-Wcast-qual)
    if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      # Clang is overeager in reporting cast alignment problems in Boost
      hpx_local_add_compile_flag_if_available(-Wcast-align)
    endif()

    # False positive when build with Vc and Clang
    if(NOT (HPXLocal_WITH_DATAPAR_VC AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL
                                         "Clang")
    )
      hpx_local_add_compile_flag_if_available(-Werror=trampolines)
    endif()
    hpx_local_add_compile_flag_if_available(-Werror=parentheses)
    hpx_local_add_compile_flag_if_available(-Werror=reorder)
    hpx_local_add_compile_flag_if_available(-Werror=return-type)
    hpx_local_add_compile_flag_if_available(-Werror=sequence-point)
    hpx_local_add_compile_flag_if_available(-Werror=uninitialized)
    hpx_local_add_compile_flag_if_available(-Werror=format)
    hpx_local_add_compile_flag_if_available(-Werror=missing-braces)
    hpx_local_add_compile_flag_if_available(-Werror=sign-compare)
  endif()
endif()

# Configure compiler warnings as errors
if(HPXLocal_WITH_COMPILER_WARNINGS_AS_ERRORS)
  if(MSVC)
    hpx_local_add_compile_flag(-WX)
  else()
    hpx_local_add_compile_flag_if_available(-Werror)
  endif()
endif()

# Diagnostics
if(MSVC)
  # Display full paths in diagnostics
  hpx_local_add_compile_flag(-FC)
  if(CMAKE_CL_64)
    set(__target_arch "x86_64")
  else()
    set(__target_arch "x86")
  endif()
  hpx_local_info("Architecture detected: ${__target_arch}")
else()
  # Show the flags that toggle each warning
  hpx_local_add_compile_flag_if_available(-fdiagnostics-show-option)

  # VLAs are a GNU extensions that we forbid as they are not supported on MSVC
  hpx_local_add_compile_flag_if_available(-Werror=vla)
  # No return statement in a non-void function can lead to garbage return values
  # in GCC.
  hpx_local_add_compile_flag_if_available(-Werror=return-type)

  # We get false positives all over the place with this.
  if(CMAKE_COMPILER_IS_GNUCXX)
    hpx_local_add_compile_flag_if_available(-Wno-unused-but-set-parameter)
    hpx_local_add_compile_flag_if_available(-Wno-unused-but-set-variable)
    # Uninitialized variables are bad, earlier compilers issue spurious warnings
    hpx_local_add_compile_flag_if_available(-Werror=uninitialized)
    hpx_local_add_compile_flag_if_available(-Wno-unused-local-typedefs)
    # -Werror=maybe-uninitialized leads to false positives.
    hpx_local_add_compile_flag_if_available(-Wno-maybe-uninitialized)
  endif()

  # False positive when building with Vc and Clang
  if(NOT (HPXLocal_WITH_DATAPAR_VC AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL
                                       "Clang")
  )
    # Silence warning about __sync_fetch_and_nand changing semantics
    hpx_local_add_compile_flag_if_available(-Wno-sync-nand)
  endif()

  # Silence warnings about deleting polymorphic objects with non-virtual dtors.
  # These come from within Boost.
  if(CMAKE_COMPILER_IS_GNUCXX)
    hpx_local_add_compile_flag_if_available(-Wno-delete-non-virtual-dtor)
  endif()

  # Check if our libraries have unresolved symbols
  if(NOT APPLE
     AND NOT WIN32
     AND NOT HPXLocal_WITH_SANITIZERS
  )
    hpx_local_add_link_flag_if_available(-Wl,-z,defs)
  endif()
  if(WIN32)
    target_link_libraries(
      hpx_local_base_libraries INTERFACE psapi WS2_32 mswsock
    )
  endif()

  if("${HPXLocal_PLATFORM_UC}" STREQUAL "BLUEGENEQ")
    hpx_local_add_compile_flag_if_available(-Wno-deprecated-register)
  endif()

  if(HPXLocal_WITH_HIDDEN_VISIBILITY)
    hpx_local_add_compile_flag_if_available(-fvisibility=hidden)
    hpx_local_add_link_flag_if_available(-fvisibility=hidden)
    hpx_local_add_config_define(HPX_HAVE_ELF_HIDDEN_VISIBILITY)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    hpx_local_add_compile_flag_if_available(-Wno-cast-align)
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # Disable the following warnings: #1170: invalid redeclaration of nested
    # class
    hpx_local_add_compile_flag_if_available(-wd1170)
    # #858: type qualifier on return type is meaningless
    hpx_local_add_compile_flag_if_available(-wd858)
    # #1098: the qualifier on this friend declaration is ignored
    hpx_local_add_compile_flag_if_available(-wd1098)
    # #488: template parameter not used in declaring the parameter type
    hpx_local_add_compile_flag_if_available(-wd488)
    # #2203: cast discards qualifiers from target type (needed for mvapich2 mpi
    # header)
    hpx_local_add_compile_flag_if_available(-wd2203)
    # #2536: cannot specify explicit initializer for arrays
    hpx_local_add_compile_flag_if_available(-wd2536)
    # #1286: invalid attribute
    hpx_local_add_compile_flag_if_available(-wd1286)
  endif()

  set(__has_timestamp_support OFF)
  include(HPXLocal_TargetArch)

  hpx_local_target_architecture(__target_arch)
  if("${__target_arch}" STREQUAL "i386"
     OR "${__target_arch}" STREQUAL "ix86"
     OR "${__target_arch}" STREQUAL "x86_64"
     OR "${__target_arch}" STREQUAL "ia64"
  )

    # rdtsc is an x86 instruction that reads the value of a CPU time stamp
    # counter. rdtscp is an extension to rdtsc. The difference is that rdtscp is
    # a serializing instruction.
    hpx_local_cpuid("rdtsc" HPXLocal_WITH_RDTSC DEFINITIONS HPX_HAVE_RDTSC)

    # One can not assume if RDTSCP is available on the hardware of the build
    # infrastructure, that it will be available on all potential target
    # hardware, see Issue  #3575
    if(NOT ${HPXLocal_WITH_BUILD_BINARY_PACKAGE})

      # XeonPhi's do not support RDTSCP
      if(NOT ("${HPXLocal_PLATFORM_UC}" STREQUAL "XEONPHI"))
        hpx_local_cpuid(
          "rdtscp" HPXLocal_WITH_RDTSCP DEFINITIONS HPX_HAVE_RDTSCP
        )
      endif()

    endif()
    if(HPXLocal_WITH_RDTSC OR HPXLocal_WITH_RDTSCP)
      set(__has_timestamp_support ON)
    endif()
  elseif(
    "${__target_arch}" STREQUAL "arm"
    OR "${__target_arch}" STREQUAL "armv5"
    OR "${__target_arch}" STREQUAL "armv6"
    OR "${__target_arch}" STREQUAL "armv7"
  )
    set(__has_timestamp_support ON)
  elseif("${__target_arch}" STREQUAL "ppc" OR "${__target_arch}" STREQUAL
                                              "ppc64"
  )
    set(__has_timestamp_support ON)
  elseif("${__target_arch}" STREQUAL "bgq")
    set(__has_timestamp_support ON)
  elseif("${__target_arch}" STREQUAL "s390x")
    set(__has_timestamp_support ON)
  endif()

  hpx_local_info("Architecture detected: ${__target_arch}")
  if(NOT __has_timestamp_support)
    hpx_local_warn(
      "No timestamp support is available; some performance counters may report incorrect results"
    )
  endif()
endif()

# store target architecture for later use
set(HPXLocal_WITH_TARGET_ARCHITECTURE
    ${__target_arch}
    CACHE INTERNAL "" FORCE
)

# Compatibility with using Boost.FileSystem, introduced in V1.4.0
set(__filesystem_compatibility_default ON)
if(HPXLocal_WITH_CXX17_FILESYSTEM)
  set(__filesystem_compatibility_default OFF)
endif()
hpx_local_option(
  HPXLocal_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY
  BOOL
  "Enable Boost.FileSystem compatibility. (default: ${__filesystem_compatibility_default})"
  ${__filesystem_compatibility_default}
  ADVANCED
  CATEGORY "Modules"
)

# Compatibility with using Boost.Iterator traversal tags, introduced in V1.7.0
hpx_local_option(
  HPXLocal_ITERATOR_SUPPORT_WITH_BOOST_ITERATOR_TRAVERSAL_TAG_COMPATIBILITY BOOL
  "Enable Boost.Iterator traversal tag compatibility. (default: OFF)" OFF
  ADVANCED CATEGORY "Modules"
)
if(HPXLocal_ITERATOR_SUPPORT_WITH_BOOST_ITERATOR_TRAVERSAL_TAG_COMPATIBILITY)
  hpx_local_add_config_define_namespace(
    DEFINE HPX_ITERATOR_SUPPORT_HAVE_BOOST_ITERATOR_TRAVERSAL_TAG_COMPATIBILITY
    NAMESPACE ITERATOR_SUPPORT
  )
endif()

# ##############################################################################
# Find Our dependencies: These are all dependencies needed to build the core
# library. Dependencies that are only needed by plugins, examples or tests
# should be found separately in the appropriate subdirectory.

include(HPXLocal_SetupThreads)
include(HPXLocal_SetupBoost)
include(HPXLocal_SetupBoostFilesystem)
include(HPXLocal_SetupAsio)
include(HPXLocal_SetupAllocator)
include(HPXLocal_SetupHwloc)
include(HPXLocal_SetupMPI)
include(HPXLocal_SetupHIP)
include(HPXLocal_SetupValgrind)

if(HPXLocal_WITH_SANITIZERS)
  hpx_local_add_config_define(HPX_HAVE_SANITIZERS)
endif()

if(HPXLocal_WITH_VIM_YCM)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# ##############################################################################
# Check Build Options based on the found dependencies. We also check for errors
# with incompatible options with the currently selected platform.
#

if(HPXLocal_WITH_GENERIC_CONTEXT_COROUTINES)
  # Check if we can use generic coroutine contexts without any problems
  if(NOT Boost_CONTEXT_FOUND)
    hpx_local_error(
      "The usage of Boost.Context was selected but Boost.Context was not found."
    )
  endif()
  hpx_local_add_config_define(HPX_HAVE_GENERIC_CONTEXT_COROUTINES)
endif()

# ##############################################################################

# ##############################################################################
# HPX_PREFIX The prefix is the default search path for HPX plugins
# ##############################################################################
if("${HPXLocal_PLATFORM_UC}" STREQUAL "ANDROID")
  set(HPX_PREFIX "lib")
  set(HPX_BUILD_PREFIX "lib")
else()
  set(HPX_PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(HPX_BUILD_PREFIX "${PROJECT_BINARY_DIR}")
endif()

# Note: on windows systems the ':' will be converted to a ';' at runtime
hpx_local_add_config_cond_define(
  HPX_DEFAULT_COMPONENT_PATH_SUFFIXES
  "\"/${CMAKE_INSTALL_LIBDIR}/hpx:/${CMAKE_INSTALL_BINDIR}/hpx:/lib/hpx:/bin/hpx\""
)

# ##############################################################################
# search path configuration
# ##############################################################################
if(HPXLocal_WITH_FULL_RPATH)
  include(HPXLocal_SetFullRPATH)
endif()

# ##############################################################################
# Git commit detection
# ##############################################################################
include(HPXLocal_GitCommit)
hpx_local_add_config_define(
  HPX_LOCAL_HAVE_GIT_COMMIT "\"${HPXLocal_WITH_GIT_COMMIT}\""
)

include(HPXLocal_SetOutputPaths)

# ##############################################################################
# Add custom targets for tests
# ##############################################################################
if(HPXLocal_WITH_TESTS)
  hpx_local_add_pseudo_target(tests)
  if(HPXLocal_WITH_TESTS_UNIT)
    hpx_local_add_pseudo_target(tests.unit)
    hpx_local_add_pseudo_dependencies(tests tests.unit)
  endif()
  if(HPXLocal_WITH_TESTS_REGRESSIONS)
    hpx_local_add_pseudo_target(tests.regressions)
    hpx_local_add_pseudo_dependencies(tests tests.regressions)
  endif()
  if(HPXLocal_WITH_TESTS_BENCHMARKS)
    hpx_local_add_pseudo_target(tests.performance)
    hpx_local_add_pseudo_dependencies(tests tests.performance)
  endif()
  if(HPXLocal_WITH_TESTS_HEADERS)
    hpx_local_add_pseudo_target(tests.headers)
    hpx_local_add_pseudo_dependencies(tests tests.headers)
  endif()
  if(HPXLocal_WITH_EXAMPLES AND HPXLocal_WITH_TESTS_EXAMPLES)
    hpx_local_add_pseudo_target(tests.examples)
    hpx_local_add_pseudo_dependencies(tests tests.examples)
  endif()

  enable_testing()
  include(CTest)
endif()

if(HPXLocal_WITH_EXAMPLES)
  hpx_local_add_pseudo_target(examples)
endif()

# ##############################################################################
# Debug library postfix
# ##############################################################################
set(CMAKE_DEBUG_POSTFIX "d")
set(HPXLocal_DEBUG_POSTFIX "d")

# ##############################################################################
# Add libraries
# ##############################################################################
add_subdirectory(libs)

# ##############################################################################
# Target specification
# ##############################################################################
if(HPXLocal_WITH_TOOLS OR HPXLocal_WITH_TESTS_BENCHMARKS)
  hpx_local_add_pseudo_target(tools)
  add_subdirectory(tools)
endif()

# ##############################################################################
# Tests
# ##############################################################################
if(HPXLocal_WITH_TESTS)
  find_package(PythonInterp)

  if(NOT PYTHONINTERP_FOUND)
    hpx_local_warn(
      "A python interpreter could not be found. The test suite can not be run automatically."
    )
  endif()

  # pseudo_target added above
  add_subdirectory(tests)
endif()

if(HPXLocal_WITH_EXAMPLES)
  add_subdirectory(examples)
endif()

# ##############################################################################
# installation instructions
# ##############################################################################
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
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPXLocal_PACKAGE_NAME}
  COMPONENT core
  PATTERN "templates" EXCLUDE
  PATTERN "packaging" EXCLUDE
  PATTERN ".git" EXCLUDE
)

# Remove all xeonphi support
if("${HPXLocal_PLATFORM_UC}" STREQUAL "XEONPHI")
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

if(HPXLocal_WITH_VIM_YCM)
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
  hpx_local_info(
    "VIM YouCompleteMe: run 'make configure_ycm' to copy config file to source directory and enable support in YCM. To enable automatic loading of configure file, add to your .vimrc option: \"let g:ycm_extra_conf_globlist = ['${PROJECT_SOURCE_DIR}/*']\""
  )
endif()

# ##############################################################################
# Add rpm packaging

hpx_local_option(
  HPXLocal_WITH_RPM BOOL "Enable or disable the generation of rpm packages" OFF
  ADVANCED
)

if(HPXLocal_WITH_RPM)
  add_subdirectory(cmake/packaging/rpm)
endif()

# ##############################################################################
# print overall configuration summary
include(HPXLocal_PrintSummary)
hpx_local_create_configuration_summary("Configuration summary:\n--" "hpxlocal")

include(HPXLocal_ExportTargets)
# Modules can't link to this if not exported
install(
  TARGETS hpx_local_base_libraries
  EXPORT HPXLocalInternalTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          COMPONENT hpx_local_base_libraries
)
hpx_local_export_internal_targets(hpx_local_base_libraries)

# ##############################################################################
# store cache vars and their values in order for them to be forwarded to the
# projects (needs to be before the HPX_GeneratePackage call)
include(HPXLocal_ForwardCacheVariables)

# ##############################################################################
# External build system support (FindHPX.cmake and pkg-config).
include(HPXLocal_GeneratePackage)

message("")
message("HPXLocal will be installed to ${CMAKE_INSTALL_PREFIX}")
message("")
