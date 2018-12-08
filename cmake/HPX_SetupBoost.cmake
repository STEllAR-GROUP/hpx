# Copyright (c) 2018 Christopher Hinz
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
    "1.66.0" "1.66"
    "1.65.0" "1.65"
    "1.64.0" "1.64"
    "1.63.0" "1.63"
    "1.62.0" "1.62"
    "1.61.0" "1.61"
    "1.60.0" "1.60"
    "1.59.0" "1.59"
    "1.58.0" "1.58"
    "1.57.0" "1.57")

set(required_Boost_components filesystem iostreams program_options system)

if(HPX_PARCELPORT_VERBS_WITH_LOGGING OR HPX_PARCELPORT_VERBS_WITH_DEV_MODE OR
        HPX_PARCELPORT_LIBFABRIC_WITH_LOGGING OR HPX_PARCELPORT_LIBFABRIC_WITH_DEV_MODE)
    set(required_Boost_components ${required_Boost_components} log log_setup date_time chrono thread)
endif()

if(HPX_WITH_THREAD_COMPATIBILITY OR NOT(HPX_WITH_CXX11_THREAD))
    set(required_Boost_components ${required_Boost_components} thread chrono)
endif()

if (HPX_WITH_TOOLS)
    set(required_Boost_components ${required_Boost_components} regex)
endif()

# Set configuration option to use Boost.Context or not. This depends on the
# platform.
set(__use_generic_coroutine_context OFF)
if(APPLE)
    set(__use_generic_coroutine_context ON)
endif()

if(HPX_PLATFORM_UC STREQUAL "BLUEGENEQ" AND Boost_VERSION GREATER 105500)
    set(__use_generic_coroutine_context ON)
endif()

hpx_option(
        HPX_WITH_GENERIC_CONTEXT_COROUTINES
        BOOL
        "Use Boost.Context as the underlying coroutines context switch implementation."
        ${__use_generic_coroutine_context} ADVANCED)

if(HPX_WITH_GENERIC_CONTEXT_COROUTINES)
    set(required_Boost_components ${required_Boost_components} context thread chrono)
endif()

find_package(Boost 1.55 REQUIRED COMPONENTS ${required_Boost_components})

add_library(hpx::boost INTERFACE IMPORTED)

# If we compile natively for the MIC, we need some workarounds for certain
# Boost headers
# FIXME: push changes upstream
if(HPX_PLATFORM_UC STREQUAL "XEONPHI")
    # This directory need to appear before the regular Boost include directories.
    set(__boost_include_dirs ${PROJECT_SOURCE_DIR}/external/asio)
endif()

set(__boost_include_dirs ${__boost_include_dirs} ${Boost_INCLUDE_DIRS})

# Emulate target_include_directories to support CMake < 3.11.
set_property(TARGET hpx::boost PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${__boost_include_dirs})
target_link_libraries(hpx::boost INTERFACE ${Boost_LIBRARIES})

# The Boost find module already links against the system thread library for versions >= 3.11.
if (CMAKE_VERSION VERSION_LESS 3.11)
    find_package(Threads REQUIRED)

    target_link_libraries(hpx::boost INTERFACE Threads::Threads)
endif()

if(NOT Boost_FOUND)
  hpx_error("Could not find Boost. Please set BOOST_ROOT to point to your Boost installation.")
endif()

# Boost preprocessor definitions
if(MSVC)
  hpx_option(HPX_WITH_BOOST_ALL_DYNAMIC_LINK BOOL
    "Add BOOST_ALL_DYN_LINK to compile flags (default: OFF)"
    OFF ADVANCED)
  if (HPX_WITH_BOOST_ALL_DYNAMIC_LINK)
    hpx_add_config_cond_define(BOOST_ALL_DYN_LINK)
  endif()
else()
  hpx_add_config_define(HPX_COROUTINE_NO_SEPARATE_CALL_SITES)
endif()
hpx_add_config_cond_define(BOOST_BIGINT_HAS_NATIVE_INT64)
