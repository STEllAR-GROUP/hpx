# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2007-2024 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_FETCH_BOOST)
  set(HPX_WITH_BOOST_VERSION "1.84.0")
  hpx_info(
    "HPX_WITH_FETCH_BOOST=${HPX_WITH_FETCH_BOOST}, Boost v${HPX_WITH_BOOST_VERSION} will be fetched using CMake's FetchContent"
  )
  include(FetchContent)
  fetchcontent_declare(
    Boost
    URL https://github.com/boostorg/boost/releases/download/boost-${HPX_WITH_BOOST_VERSION}/boost-${HPX_WITH_BOOST_VERSION}.tar.gz
    TLS_VERIFY true
    DOWNLOAD_EXTRACT_TIMESTAMP true
  )
  fetchcontent_populate(Boost)
  set(HPX_WITH_BUILD_FETCHED_BOOST
      "Execute process"
      CACHE STRING "Used by command line tool to build fetched Boost"
  )
  set(HPX_WITH_BUILD_FETCHED_BOOST_CHECK
      ""
      CACHE
        STRING
        "for internal use only, do not modify. Checks if fetched Boost is built"
  )

  if(NOT HPX_WITH_BUILD_FETCHED_BOOST STREQUAL
     HPX_WITH_BUILD_FETCHED_BOOST_CHECK
  )
    if(WIN32)
      execute_process(
        COMMAND
          cmd /C
          "cd ${CMAKE_BINARY_DIR}\\_deps\\boost-src && .\\bootstrap.bat && .\\b2 headers cxxflags=/std:c++${HPX_CXX_STANDARD}"
      )
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      execute_process(
        COMMAND
          sh -c
          "cd ${CMAKE_BINARY_DIR}/_deps/boost-src && ./bootstrap.sh --prefix=${CMAKE_BINARY_DIR}/_deps/boost-installed && ./b2 && ./b2 install --prefix=${CMAKE_BINARY_DIR}/_deps/boost-installed cxxflags=--std=c++${HPX_CXX_STANDARD}"
      )
    else()
      execute_process(
        COMMAND
          sh -c
          "cd ${CMAKE_BINARY_DIR}/_deps/boost-src && ./bootstrap.sh && ./b2 headers cxxflags=--std=c++${HPX_CXX_STANDARD}"
      )
    endif()
    set(HPX_WITH_BUILD_FETCHED_BOOST_CHECK
        ${HPX_WITH_BUILD_FETCHED_BOOST}
        CACHE
          INTERNAL
          "for internal use only, do not modify. Checks if fetched Boost is built"
    )
  endif()

  set(Boost_DIR
      "${CMAKE_BINARY_DIR}/_deps/boost-src"
      CACHE INTERNAL ""
  )
  set(Boost_INCLUDE_DIR
      "${CMAKE_BINARY_DIR}/_deps/boost-src"
      CACHE INTERNAL ""
  )
endif()

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
      "1.85.0" "1.85"
      "1.84.0" "1.84"
      "1.83.0" "1.83"
      "1.82.0" "1.82"
      "1.81.0" "1.81"
      "1.80.0" "1.80"
      "1.79.0" "1.79"
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

  hpx_set_cmake_policy(CMP0167 OLD) # use CMake's FindBoost for now

  # Find the headers and get the version
  find_package(Boost ${Boost_MINIMUM_VERSION} NO_POLICY_SCOPE MODULE REQUIRED)
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

  # compatibility with older CMake versions
  if(BOOST_ROOT AND NOT Boost_ROOT)
    set(Boost_ROOT
        ${BOOST_ROOT}
        CACHE PATH "Boost base directory"
    )
    unset(BOOST_ROOT CACHE)
  endif()

  find_package(
    Boost ${Boost_MINIMUM_VERSION} NO_POLICY_SCOPE MODULE REQUIRED
    COMPONENTS ${__boost_libraries}
  )

  if(NOT Boost_FOUND)
    hpx_error(
      "Could not find Boost. Please set Boost_ROOT to point to your Boost installation."
    )
  endif()

  # We are assuming that there is only one Boost Root
  if(NOT Boost_ROOT AND "$ENV{BOOST_ROOT}")
    set(Boost_ROOT $ENV{BOOST_ROOT})
  elseif(NOT Boost_ROOT)
    string(REPLACE "/include" "" Boost_ROOT "${Boost_INCLUDE_DIRS}")
  endif()

  if(Boost_ROOT)
    file(TO_CMAKE_PATH ${Boost_ROOT} Boost_ROOT)
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
