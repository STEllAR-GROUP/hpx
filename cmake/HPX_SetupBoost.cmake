# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We first try to find the required minimum set of Boost libraries. This will
# also give us the version of the found boost installation
if(HPX_WITH_STATIC_LINKING)
  set(Boost_USE_STATIC_LIBS ON)
endif()

# Add additional version to recognize
set(Boost_ADDITIONAL_VERSIONS
    ${Boost_ADDITIONAL_VERSIONS}
    "1.71.0" "1.71"
    "1.70.0" "1.70"
    "1.69.0" "1.69"
    "1.68.0" "1.68"
    "1.67.0" "1.67"
    "1.66.0" "1.66"
    "1.65.0" "1.65" "1.65.1" "1.65.1"
    "1.64.0" "1.64"
    "1.63.0" "1.63"
    "1.62.0" "1.62"
    "1.61.0" "1.61"
    "1.60.0" "1.60"
    "1.59.0" "1.59"
    "1.58.0" "1.58"
    "1.57.0" "1.57")
set(Boost_MINIMUM_VERSION "1.61" CACHE  INTERNAL "1.61" FORCE)

set(__boost_libraries)
if(HPX_PARCELPORT_VERBS_WITH_LOGGING OR HPX_PARCELPORT_VERBS_WITH_DEV_MODE OR
   HPX_PARCELPORT_LIBFABRIC_WITH_LOGGING OR HPX_PARCELPORT_LIBFABRIC_WITH_DEV_MODE)
  set(__boost_libraries ${__boost_libraries} log log_setup date_time chrono thread)
endif()

# Set configuration option to use Boost.Context or not. This depends on the
# platform.
set(__use_generic_coroutine_context OFF)
if(APPLE)
  set(__use_generic_coroutine_context ON)
endif()
if(HPX_PLATFORM_UC STREQUAL "BLUEGENEQ")
  set(__use_generic_coroutine_context ON)
endif()
hpx_option(
  HPX_WITH_GENERIC_CONTEXT_COROUTINES
  BOOL
  "Use Boost.Context as the underlying coroutines context switch implementation."
  ${__use_generic_coroutine_context} ADVANCED)

if(NOT HPX_WITH_NATIVE_TLS)
  set(__boost_libraries ${__boost_libraries} thread)
endif()

if(HPX_WITH_GENERIC_CONTEXT_COROUTINES)
  set(__boost_libraries ${__boost_libraries} context)
  # if context is needed, we should still link with boost thread and chrono
  set(__boost_libraries ${__boost_libraries} thread chrono)
endif()

set(__boost_libraries
  ${__boost_libraries}
  system)

set(Boost_NO_BOOST_CMAKE ON) # disable the search for boost-cmake
find_package(Boost ${Boost_MINIMUM_VERSION}
  MODULE REQUIRED
  COMPONENTS ${__boost_libraries})

if(NOT Boost_FOUND)
  hpx_error("Could not find Boost. Please set BOOST_ROOT to point to your Boost installation.")
endif()

set(Boost_TMP_LIBRARIES ${Boost_LIBRARIES})
if(UNIX AND NOT CYGWIN)
  find_library(BOOST_UNDERLYING_THREAD_LIBRARY NAMES pthread DOC "The threading library used by boost.thread")
  if(NOT BOOST_UNDERLYING_THREAD_LIBRARY AND (HPX_PLATFORM_UC STREQUAL "XEONPHI"))
    set(BOOST_UNDERLYING_THREAD_LIBRARY "-pthread")
  endif()
  set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${BOOST_UNDERLYING_THREAD_LIBRARY})
endif()

set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${Boost_LIBRARIES})

if(NOT HPX_WITH_CXX17_FILESYSTEM)
  find_package(Boost ${Boost_MINIMUM_VERSION}
    QUIET MODULE
    COMPONENTS filesystem)
  if(Boost_FILESYSTEM_FOUND)
    hpx_info("  filesystem")
  else()
    hpx_error("Could not find Boost.Filesystem. Either use a compiler with support for C++17 filesystem or provide a boost installation including the filesystem library")
  endif()
  set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${Boost_LIBRARIES})
endif()

if(HPX_WITH_COMPRESSION_BZIP2 OR HPX_WITH_COMPRESSION_ZLIB)
  find_package(Boost 1.61 QUIET MODULE COMPONENTS iostreams)
  if(Boost_IOSTREAMS_FOUND)
    hpx_info("  iostreams")
  else()
    hpx_error("Could not find Boost.Iostreams but HPX_WITH_COMPRESSION_BZIP2=On or HPX_WITH_COMPRESSION_LIB=On. Either set it to off or provide a boost installation including the iostreams library")
  endif()
  set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${Boost_LIBRARIES})
endif()

if(HPX_WITH_TOOLS)
  find_package(Boost 1.61 QUIET MODULE COMPONENTS regex)
  if(Boost_REGEX_FOUND)
    hpx_info("  regex")
  else()
    hpx_error("Could not find Boost.Regex but HPX_WITH_TOOLS=On (the inspect tool requires Boost.Regex). Either set it to off or provide a boost installation including the regex library")
  endif()
  set(Boost_TMP_LIBRARIES ${Boost_TMP_LIBRARIES} ${Boost_LIBRARIES})
endif()

set(Boost_LIBRARIES ${Boost_TMP_LIBRARIES})

# If we compile natively for the MIC, we need some workarounds for certain
# Boost headers
# FIXME: push changes upstream
if(HPX_PLATFORM_UC STREQUAL "XEONPHI")
  set(Boost_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/asio ${Boost_INCLUDE_DIRS})
endif()

# Boost preprocessor definitions
hpx_add_config_cond_define(BOOST_PARAMETER_MAX_ARITY 7)
if(NOT Boost_USE_STATIC_LIBS)
  hpx_add_config_cond_define(BOOST_ALL_DYN_LINK)
endif()
if(NOT MSVC)
  hpx_add_config_define(HPX_COROUTINE_NO_SEPARATE_CALL_SITES)
endif()
hpx_add_config_cond_define(BOOST_BIGINT_HAS_NATIVE_INT64)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
add_definitions(-DBOOST_ALL_NO_LIB) # disable auto-linking
hpx_libraries(${Boost_LIBRARIES})
