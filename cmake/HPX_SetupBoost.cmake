# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
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

set(Boost_NO_BOOST_CMAKE ON) # disable the search for boost-cmake

# Find the headers and get the version
find_package(Boost ${Boost_MINIMUM_VERSION} REQUIRED)
if(NOT Boost_VERSION_STRING)
  set(Boost_VERSION_STRING "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
endif()

set(__boost_libraries "")
if(HPX_PARCELPORT_VERBS_WITH_LOGGING OR HPX_PARCELPORT_VERBS_WITH_DEV_MODE OR
   HPX_PARCELPORT_LIBFABRIC_WITH_LOGGING OR HPX_PARCELPORT_LIBFABRIC_WITH_DEV_MODE)
  set(__boost_libraries ${__boost_libraries} log log_setup date_time chrono thread)
endif()

# Boost.System is header-only from 1.69 onwards. But filesystem is needed the
# libboost_system.so library in Boost 1.69 so can't link only to filesystem
if(Boost_VERSION_STRING VERSION_LESS 1.70)
  set(__boost_libraries ${__boost_libraries} system)
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
  if(CMAKE_VERSION VERSION_LESS 3.12)
    hpx_error("The Boost.context component needs at least CMake 3.12.3 to be \
    found.")
  endif()
  # if context is needed, we should still link with boost thread and chrono
  set(__boost_libraries ${__boost_libraries} context thread chrono)
endif()

list(REMOVE_DUPLICATES __boost_libraries)

find_package(Boost ${Boost_MINIMUM_VERSION}
  MODULE REQUIRED
  COMPONENTS ${__boost_libraries})

if(NOT Boost_FOUND)
  hpx_error("Could not find Boost. Please set BOOST_ROOT to point to your Boost installation.")
endif()

# We are assuming that there is only one Boost Root
if (NOT BOOST_ROOT AND "$ENV{BOOST_ROOT}")
  set(BOOST_ROOT $ENV{BOOST_ROOT})
elseif(NOT BOOST_ROOT)
  string(REPLACE "/include" "" BOOST_ROOT "${Boost_INCLUDE_DIRS}")
endif()

add_library(hpx::boost INTERFACE IMPORTED)

# If we compile natively for the MIC, we need some workarounds for certain
# Boost headers
# FIXME: push changes upstream
if(HPX_PLATFORM_UC STREQUAL "XEONPHI")
  # Before flag remove when passing at set_property for cmake < 3.11 instead of target_include_directories
  # so should be added first
  set_property(TARGET hpx::boost PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/external/asio)
endif()

set_property(TARGET hpx::boost APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
  set_property(TARGET hpx::boost APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
else()
  target_link_libraries(hpx::boost INTERFACE ${Boost_LIBRARIES})
endif()

find_package(Threads QUIET REQUIRED)
if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
  set_property(TARGET hpx::boost APPEND PROPERTY INTERFACE_LINK_LIBRARIES Threads::Threads)
else()
  target_link_libraries(hpx::boost INTERFACE Threads::Threads)
endif()

# If we compile natively for the MIC, we need some workarounds for certain
# Boost headers
# FIXME: push changes upstream
if(HPX_PLATFORM_UC STREQUAL "XEONPHI")
  target_include_directories(hpx::boost BEFORE INTERFACE ${PROJECT_SOURCE_DIR}/external/asio)
endif()

include(HPX_AddDefinitions)
# Boost preprocessor definitions
if(NOT Boost_USE_STATIC_LIBS)
  hpx_add_config_cond_define(BOOST_ALL_DYN_LINK)
endif()
if(NOT MSVC)
  hpx_add_config_define(HPX_COROUTINE_NO_SEPARATE_CALL_SITES)
endif()
hpx_add_config_cond_define(BOOST_BIGINT_HAS_NATIVE_INT64)
set_property(TARGET hpx::boost APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS BOOST_ALL_NO_LIB) # disable auto-linking
