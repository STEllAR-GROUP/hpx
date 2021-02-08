# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# In case find_package(HPX) is called multiple times
if(NOT TARGET hpx_dependencies_boost)
  # We first try to find the required minimum set of Boost libraries. This will
  # also give us the version of the found boost installation
  if(HPX_WITH_STATIC_LINKING)
    set(Boost_USE_STATIC_LIBS ON)
  endif()

  # Add additional version to recognize
  # cmake-format: off
  set(Boost_ADDITIONAL_VERSIONS
      ${Boost_ADDITIONAL_VERSIONS}
      "1.75.0" "1.75"
      "1.74.0" "1.74"
      "1.73.0" "1.73"
      "1.72.0" "1.72"
      "1.71.0" "1.71"
      "1.70.0" "1.70"
      "1.69.0" "1.69"
      "1.68.0" "1.68"
      "1.67.0" "1.67"
      "1.66.0" "1.66"
  )
  # cmake-format: on
  set(Boost_MINIMUM_VERSION
      "1.66"
      CACHE INTERNAL "1.66" FORCE
  )

  set(Boost_NO_BOOST_CMAKE ON) # disable the search for boost-cmake

  # Find the headers and get the version
  find_package(Boost ${Boost_MINIMUM_VERSION} REQUIRED)
  if(NOT Boost_VERSION_STRING)
    set(Boost_VERSION_STRING
        "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}"
    )
  endif()

  set(__boost_libraries "")
  if(HPX_PARCELPORT_LIBFABRIC_WITH_LOGGING
     OR HPX_PARCELPORT_LIBFABRIC_WITH_DEV_MODE
  )
    set(__boost_libraries ${__boost_libraries} log log_setup date_time chrono
                          thread
    )
  endif()

  # Boost.System is header-only from 1.69 onwards. But filesystem is needed the
  # libboost_system.so library in Boost 1.69 so can't link only to filesystem
  if(Boost_VERSION_STRING VERSION_LESS 1.70)
    set(__boost_libraries ${__boost_libraries} system)
  endif()

  if(NOT HPX_WITH_NATIVE_TLS)
    set(__boost_libraries ${__boost_libraries} thread)
  endif()

  if(HPX_WITH_GENERIC_CONTEXT_COROUTINES)
    # if context is needed, we should still link with boost thread and chrono
    set(__boost_libraries ${__boost_libraries} context thread chrono)
  endif()

  list(REMOVE_DUPLICATES __boost_libraries)

  find_package(
    Boost ${Boost_MINIMUM_VERSION} MODULE REQUIRED
    COMPONENTS ${__boost_libraries}
  )

  if(NOT Boost_FOUND)
    hpx_error(
      "Could not find Boost. Please set BOOST_ROOT to point to your Boost installation."
    )
  endif()

  # target_arch should be set already by the parent CMakeLists file
  if("${__target_arch}" STREQUAL "ppc" OR "${__target_arch}" STREQUAL "ppc64")
    if("${Boost_VERSION_STRING}" VERSION_LESS "1.69.0")
      hpx_error("Boost versions < 1.69.0 are unsupported on PPC architecture")
    endif()
  endif()

  # We are assuming that there is only one Boost Root
  if(NOT BOOST_ROOT AND "$ENV{BOOST_ROOT}")
    set(BOOST_ROOT $ENV{BOOST_ROOT})
  elseif(NOT BOOST_ROOT)
    string(REPLACE "/include" "" BOOST_ROOT "${Boost_INCLUDE_DIRS}")
  endif()

  add_library(hpx_dependencies_boost INTERFACE IMPORTED)

  # If we compile natively for the MIC, we need some workarounds for certain
  # Boost headers FIXME: push changes upstream
  if(HPX_PLATFORM_UC STREQUAL "XEONPHI")
    target_include_directories(
      hpx_dependencies_boost SYSTEM BEFORE
      INTERFACE ${PROJECT_SOURCE_DIR}/external/asio
    )
  endif()

  target_link_libraries(hpx_dependencies_boost INTERFACE Boost::boost)
  foreach(__boost_library ${__boost_libraries})
    target_link_libraries(
      hpx_dependencies_boost INTERFACE Boost::${__boost_library}
    )
  endforeach()

  include(HPX_AddDefinitions)

  # Boost Asio should not use Boost exceptions
  hpx_add_config_cond_define(BOOST_ASIO_HAS_BOOST_THROW_EXCEPTION 0)
  # Disable concepts support in Asio as a workaround to
  # https://github.com/boostorg/asio/issues/312
  hpx_add_config_cond_define(BOOST_ASIO_DISABLE_CONCEPTS)
  # Disable experimental std::string_view support as a workaround to
  # https://github.com/chriskohlhoff/asio/issues/597
  hpx_add_config_cond_define(BOOST_ASIO_DISABLE_STD_EXPERIMENTAL_STRING_VIEW)
  if(Boost_VERSION_STRING VERSION_LESS 1.67)
    hpx_add_config_cond_define(BOOST_ASIO_DISABLE_STD_STRING_VIEW)
  endif()

  find_package(Threads QUIET REQUIRED)
  target_link_libraries(hpx_dependencies_boost INTERFACE Threads::Threads)

  # Boost preprocessor definitions
  if(NOT Boost_USE_STATIC_LIBS)
    hpx_add_config_cond_define(BOOST_ALL_DYN_LINK)
  endif()
  if(NOT MSVC)
    hpx_add_config_define(HPX_COROUTINE_NO_SEPARATE_CALL_SITES)
  endif()
  hpx_add_config_cond_define(BOOST_BIGINT_HAS_NATIVE_INT64)
  target_link_libraries(
    hpx_dependencies_boost INTERFACE Boost::disable_autolinking
  )
endif()
