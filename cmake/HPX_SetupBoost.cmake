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
      "1.78.0" "1.78"
      "1.77.0" "1.77"
      "1.76.0" "1.76"
      "1.75.0" "1.75"
      "1.74.0" "1.74"
      "1.73.0" "1.73"
      "1.72.0" "1.72"
      "1.71.0" "1.71"
  )
  # cmake-format: on
  set(Boost_MINIMUM_VERSION
      "1.71"
      CACHE INTERNAL "1.71" FORCE
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

  # We are assuming that there is only one Boost Root
  if(NOT BOOST_ROOT AND "$ENV{BOOST_ROOT}")
    set(BOOST_ROOT $ENV{BOOST_ROOT})
  elseif(NOT BOOST_ROOT)
    string(REPLACE "/include" "" BOOST_ROOT "${Boost_INCLUDE_DIRS}")
  endif()

  add_library(hpx_dependencies_boost INTERFACE IMPORTED)

  target_link_libraries(hpx_dependencies_boost INTERFACE Boost::boost)
  foreach(__boost_library ${__boost_libraries})
    target_link_libraries(
      hpx_dependencies_boost INTERFACE Boost::${__boost_library}
    )
  endforeach()

  if(HPX_WITH_HIP AND Boost_VERSION VERSION_LESS 1.78)
    target_compile_definitions(
      hpx_dependencies_boost
      INTERFACE "BOOST_NOINLINE=__attribute__ ((noinline))"
    )
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
  target_link_libraries(
    hpx_dependencies_boost INTERFACE Boost::disable_autolinking
  )
endif()
