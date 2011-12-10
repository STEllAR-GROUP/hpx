# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We require at least CMake V2.8.4
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

################################################################################
# Build type 
################################################################################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Configuration type" FORCE)
endif()

set(CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPE}" CACHE STRING "Configuration types" FORCE)

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
# project metadata
################################################################################
project(hpx CXX)

set(HPX_MAJOR_VERSION 1)
set(HPX_MINOR_VERSION 0)
set(HPX_PATCH_LEVEL   0)
set(HPX_VERSION "${HPX_MAJOR_VERSION}.${HPX_MINOR_VERSION}.${HPX_PATCH_LEVEL}")
set(HPX_SOVERSION ${HPX_MAJOR_VERSION})

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

################################################################################
# Fortran compiler detection
################################################################################
include(HPX_FortranCompiler)

################################################################################
# search path configuration
################################################################################
include_directories(${hpx_SOURCE_DIR})
include_directories(${hpx_SOURCE_DIR}/external/move)
include_directories(${hpx_SOURCE_DIR}/external/atomic)
include_directories(${hpx_SOURCE_DIR}/external/cache)
include_directories(${hpx_SOURCE_DIR}/external/coroutine)
include_directories(${hpx_SOURCE_DIR}/external/endian)
include_directories(${hpx_SOURCE_DIR}/external/logging)
include_directories(${hpx_SOURCE_DIR}/external/lockfree)
include_directories(${hpx_SOURCE_DIR}/external/plugin)
include_directories(${hpx_SOURCE_DIR}/external/backtrace)
include_directories(${hpx_SOURCE_DIR}/external/serialization)
link_directories(${CMAKE_BINARY_DIR}/lib/hpx)

################################################################################
# Boost configuration
################################################################################
# Boost.Chrono is in the Boost trunk, but has not been in a Boost release yet

hpx_option(HPX_INTERNAL_CHRONO BOOL "Use HPX's internal version of Boost.Chrono (default: ON)" ON)

# this cmake module will snag the Boost version we'll be using (which we need
# to know to specify the Boost libraries that we want to look for).
find_package(HPX_BoostVersion)

if(NOT BOOST_VERSION_FOUND)
  hpx_error("boost" "Failed to locate Boost.")
endif()

if(NOT MSVC AND ${BOOST_MINOR_VERSION} LESS 47)
  hpx_warn("boost" "Applications will segfault at shutdown with this version of Boost, please use 1.47.0 or higher.")
endif()

if(NOT HPX_INTERNAL_CHRONO OR ${BOOST_MINOR_VERSION} GREATER 46)
  set(BOOST_LIBRARIES chrono
                      date_time
                      filesystem
                      program_options
                      regex
                      serialization
                      system
                      thread)
else()
  set(BOOST_LIBRARIES date_time
                      filesystem
                      program_options
                      regex
                      serialization
                      system
                      thread)
  add_definitions(-DHPX_INTERNAL_CHRONO)
  add_definitions(-DBOOST_CHRONO_NO_LIB)
  include_directories(${hpx_SOURCE_DIR}/external/chrono)
endif()

# We have a patched version of FindBoost loosely based on the one that Kitware ships
find_package(HPX_Boost)

include_directories(${BOOST_INCLUDE_DIR})
link_directories(${BOOST_LIBRARY_DIR})

# Boost preprocessor definitions
add_definitions(-DBOOST_PARAMETER_MAX_ARITY=7)
add_definitions(-DBOOST_COROUTINE_USE_ATOMIC_COUNT)
add_definitions(-DBOOST_COROUTINE_ARG_MAX=2)
add_definitions(-DBOOST_LOG_NO_TSS)
add_definitions(-DBOOST_LOG_NO_TS)
add_definitions(-DBOOST_BIGINT_HAS_NATIVE_INT64)
add_definitions(-DBOOST_NO_0X_HDR_ATOMIC)        # disable usage of std::atomics in lockfree

################################################################################
# Compiler detection code
################################################################################

# C++
hpx_include(GCCVersion)

if(GCC_VERSION)
  hpx_info("gcc_config" "Compiler reports compatibility with GCC version ${GCC_VERSION_STR}")

  if(040400 GREATER ${GCC_VERSION})
    hpx_error("gcc_config" "GCC 4.4.x or higher is required.")
  endif()
elseif(NOT MSVC10 OR NOT CMAKE_CL_64)
  hpx_error("msvc_config" "MSVC x64 2010 or higher is required.")
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
# installation configuration
################################################################################
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

add_definitions("-DHPX_PREFIX=\"${CMAKE_INSTALL_PREFIX}\"")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

################################################################################
# RPATH configuration
################################################################################
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib/hpx:${CMAKE_BINARY_DIR}/bin")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

################################################################################
# C++11 configuration
################################################################################
hpx_option(HPX_CXX11 BOOL "Use C++11 support (default: ON)" ON)

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
# HDF5 output
################################################################################
find_package(HPX_HDF5)

################################################################################
# Warning configuration
################################################################################
hpx_option(HPX_WARNINGS BOOL "Enable compiler warnings (default: ON)" ON)

################################################################################
# Backtrace configuration
################################################################################
hpx_option(HPX_STACKTRACES BOOL "Attach backtraces to HPX exceptions (default: ON)" ON)

if(HPX_STACKTRACES)
  hpx_info("stacktraces" "Stack traces are enabled.")
  add_definitions(-DHPX_HAVE_STACKTRACES)
endif()

################################################################################
# Native TLS configuration
################################################################################
hpx_option(HPX_NATIVE_TLS BOOL "Use native TLS support if available (default: ON)" ON)

if(HPX_NATIVE_TLS)
  hpx_info("tls" "Native TLS is enabled.")
  add_definitions(-DHPX_HAVE_NATIVE_TLS)
endif()

################################################################################
# pxthread stack size
################################################################################
if(NOT HPX_STACK_SIZE)
  if(MSVC)
    set(HPX_STACK_SIZE "0x4000" CACHE STRING "pxthread stack size" FORCE)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR
     "CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    set(HPX_STACK_SIZE "0x10000" CACHE STRING "pxthread stack size" FORCE)
  else()
    set(HPX_STACK_SIZE "0x8000" CACHE STRING "pxthread stack size" FORCE)
  endif()
else()
  set(HPX_STACK_SIZE "${HPX_STACK_SIZE}" CACHE STRING "pxthread stack size" FORCE)
endif()

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

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DDEBUG)
  add_definitions(-DHPX_BUILD_TYPE="debug")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
  add_definitions(-DDEBUG)
  add_definitions(-DHPX_BUILD_TYPE="relwithdebinfo")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
  add_definitions(-DNDEBUG)
  add_definitions(-DHPX_BUILD_TYPE="minsizerel")
elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  add_definitions(-DNDEBUG)
  add_definitions(-DBOOST_DISABLE_ASSERTS)
  add_definitions(-DHPX_BUILD_TYPE="release")
else()
  hpx_error("build_type" "\"${CMAKE_BUILD_TYPE}\" is not a valid build type.")
endif()

# Debug library postfix
set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "d")

################################################################################
# Utility configuration 
################################################################################
hpx_option(HPX_UTIL_BIND BOOL "Use hpx::util::bind instead of boost::bind or std::bind (default: ON)" ON)

if(HPX_UTIL_BIND)
  add_definitions(-DHPX_UTIL_BIND)
endif()

hpx_option(HPX_UTIL_FUNCTION BOOL "Use hpx::util::function instead of boost::function or std::function (default: ON)" ON)

if(HPX_USE_UTIL_FUNCTION)
  add_definitions(-DHPX_UTIL_FUNCTION)
endif()

################################################################################
# Scheduler configuration
################################################################################
hpx_option(HPX_GLOBAL_SCHEDULER BOOL "Enable the use of --queueing=global (default: OFF)" OFF)

if(HPX_GLOBAL_SCHEDULER)
  add_definitions(-DHPX_GLOBAL_SCHEDULER)
endif()

hpx_option(HPX_LOCAL_SCHEDULER BOOL "Enable the use of --queueing=global (default: OFF)" OFF)

if(HPX_LOCAL_SCHEDULER)
  add_definitions(-DHPX_LOCAL_SCHEDULER)
endif()

hpx_option(HPX_ABP_SCHEDULER BOOL "Enable the use of --queueing=abp (default: OFF)" OFF)

if(HPX_ABP_SCHEDULER)
  add_definitions(-DHPX_ABP_SCHEDULER)
endif()

################################################################################
# MSVC configuration
################################################################################
if(MSVC)
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    # No optimizations
    hpx_append_flag(-Od LANGUAGES CXX Fortran)

    # Generate debugging information
    hpx_append_flag(-Zi LANGUAGES CXX Fortran)

    # Keep the frame pointer
    hpx_append_flag(-Oy- LANGUAGES CXX Fortran)

    # Don't inline functions
    hpx_append_flag(-Ob0 LANGUAGES CXX Fortran)

    # Use multi-threaded debug runtime
    hpx_append_flag(-MDd LANGUAGES CXX Fortran)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    # Optimize for speed
    hpx_append_flag(-O2 LANGUAGES CXX Fortran)

    # Generate debugging information
    hpx_append_flag(-Zi LANGUAGES CXX Fortran)

    # Keep the frame pointer
    hpx_append_flag(-Oy- LANGUAGES CXX Fortran)

    # Don't inline functions
    hpx_append_flag(-Ob0 LANGUAGES CXX Fortran)

    # Use multi-threaded debug runtime
    hpx_append_flag(-MDd LANGUAGES CXX Fortran)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    # Optimize for size
    hpx_append_flag(-O1 LANGUAGES CXX Fortran)

    # Use multi-threaded runtime
    hpx_append_flag(-MD LANGUAGES CXX Fortran)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    # Optimize for speed
    hpx_append_flag(-O2 LANGUAGES CXX Fortran)

    # Use multi-threaded runtime
    hpx_append_flag(-MD LANGUAGES CXX Fortran)
  endif()

  # Exceptions
  hpx_append_flag(-EHsc)

  # Runtime type information
  hpx_append_flag(-GR)

  # Multiproccessor build
  hpx_append_flag(-MP)

  # Increase the maximum size of object file sections
  hpx_append_flag(-bigobj LANGUAGES CXX Fortran)

  set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} psapi)

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
    set(HPX_CXX11_RVALUE_REFERENCES ON CACHE STRING "cxx11_rvalue_references state.")
    add_definitions(-DHPX_HAVE_CXX11_RVALUE_REFERENCES)

    set(HPX_CXX11_LAMBDAS ON CACHE STRING "cxx11_lambdas state.")
    add_definitions(-DHPX_HAVE_CXX11_LAMBDAS)

    set(HPX_CXX11_AUTO ON CACHE STRING "cxx11_auto state.")
    add_definitions(-DHPX_HAVE_CXX11_AUTO)

    set(HPX_CXX11_UNIQUE_PTR ON CACHE STRING "cxx11_std_unique_ptr state.")
    add_definitions(-DHPX_HAVE_CXX11_STD_UNIQUE_PTR)
    add_definitions(-DBOOST_LOCKFREE_HAVE_CXX11_UNIQUE_PTR)
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
  add_definitions(-D_WIN32_WINNT=0x0600)
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

    # Keep the frame pointer
    hpx_use_flag_if_available(-fno-omit-frame-pointer LANGUAGES CXX Fortran)

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

    hpx_check_for_cxx11_std_bind(HPX_CXX11_STD_BIND
      DEFINITIONS HPX_HAVE_CXX11_STD_BIND)

    hpx_check_for_cxx11_std_function(HPX_CXX11_STD_FUNCTION
      DEFINITIONS HPX_HAVE_CXX11_STD_FUNCTION)
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

  # Uninitialized variables are bad
  hpx_use_flag_if_available(-Werror=uninitialized)

  # Show the flags that toggle each warning
  hpx_use_flag_if_available(-fdiagnostics-show-option LANGUAGES CXX Fortran)

  # We get false positives all over the place with this. Also, detection for
  # this flag fails with GCC 4.4 and 4.5.
  if(040600 LESS ${GCC_VERSION} OR 040600 EQUAL ${GCC_VERSION})
    hpx_use_flag_if_available(-Wno-unused-but-set-parameter)
    hpx_use_flag_if_available(-Wno-unused-but-set-variable)
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
           "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
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
  set(hpx_RUNTIME_LIBRARIES dl rt)

  hpx_check_for_pthread_affinity_np(HPX_PTHREAD_AFFINITY_NP
    DEFINITIONS HPX_HAVE_PTHREAD_AFFINITY_NP)

  hpx_use_flag_if_available(-pthread)

  set(HPX_GOOGLE_PERFTOOLS ON CACHE STRING "Compile HPX with google perftools" FORCE)

  if(HPX_GOOGLE_PERFTOOLS)
    find_package(HPX_GooglePerftools)
    if(GOOGLE_PERFTOOLS_FOUND AND HPX_GOOGLE_PERFTOOLS)
      set(hpx_LIBRARIES ${hpx_LIBRARIES} ${GOOGLE_PERFTOOLS_LIBRARY})
      include_directories(${GOOGLE_PERFTOOLS_INCLUDE_DIR})
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
        add_definitions(-DHPX_TCMALLOC)
      else()
        hpx_info("malloc" "Using jemalloc allocator.")
        set(hpx_MALLOC_LIBRARY ${JEMALLOC_LIBRARY})
        set(hpx_LIBRARIES ${hpx_LIBRARIES} ${JEMALLOC_LIBRARY})
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

option(HPX_EXAMPLES "Build HPX examples (default: ON)" ON)

if(HPX_EXAMPLES)
  add_hpx_pseudo_target(examples)

  include_directories(examples)
  add_subdirectory(examples)
endif()

option(HPX_TESTS "Build HPX tests (default: ON)" ON)

if(HPX_TESTS)
  add_hpx_pseudo_target(tests)

  include_directories(tests)
  add_subdirectory(tests)
endif()

option(HPX_RUNTIME "Build HPX runtime (default: ON)" ON)

if(HPX_RUNTIME)
  add_hpx_pseudo_target(runtime)

  add_subdirectory(runtime)
endif()

option(HPX_TOOLS "Build HPX tools (default: OFF)" OFF)

if(HPX_TOOLS)
  add_hpx_pseudo_target(tools)

  add_subdirectory(tools)
endif()

################################################################################
# installation instructions
################################################################################
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
  DESTINATION share/hpx/cmake
  COMPONENT core
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
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
  PATTERN ".git" EXCLUDE)

if(HPX_INTERNAL_CHRONO AND ${BOOST_MINOR_VERSION} LESS 47)
  install( # install Boost.Chrono headers if we're using our internal version
    DIRECTORY external/chrono/boost
    DESTINATION include
    COMPONENT core
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN ".svn" EXCLUDE
    PATTERN ".git" EXCLUDE)
endif()

