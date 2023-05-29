# Copyright (c) 2023 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_FETCH_JSON)
  find_package(nlohmann_json 3.2.0 REQUIRED)
elseif(NOT TARGET JSON::json)
  if(NOT HPX_WITH_JSON_TAG)
    set(HPX_WITH_JSON_TAG "v3.11.2")
  endif()

  if(FETCHCONTENT_SOURCE_DIR_JSON)
    hpx_info(
      "HPX_WITH_FETCH_JSON=${HPX_WITH_FETCH_JSON}, JSON will be used through CMake's FetchContent and installed alongside HPX (FETCHCONTENT_SOURCE_DIR_JSON=${FETCHCONTENT_SOURCE_DIR_JSON})"
    )
  else()
    hpx_info(
      "HPX_WITH_FETCH_JSON=${HPX_WITH_FETCH_JSON}, JSON will be fetched using CMake's FetchContent and installed alongside HPX (HPX_WITH_JSON_TAG=${HPX_WITH_JSON_TAG})"
    )
  endif()

  include(FetchContent)
  fetchcontent_declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG ${HPX_WITH_JSON_TAG}
  )
  fetchcontent_makeavailable(nlohmann_json)

  set(JSON_ROOT ${nlohmann_json_SOURCE_DIR})

  add_library(json INTERFACE)
  target_include_directories(
    json SYSTEM INTERFACE $<BUILD_INTERFACE:${JSON_ROOT}/include>
                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  target_compile_definitions(json INTERFACE JSON_HAS_CPP_17)

  install(
    TARGETS json
    EXPORT HPXJSONTarget
    COMPONENT core
  )

  install(
    DIRECTORY ${JSON_ROOT}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT core
    FILES_MATCHING
    PATTERN "*.hpp"
  )

  export(
    TARGETS json
    NAMESPACE JSON::
    FILE "${CMAKE_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXJSONTarget.cmake"
  )

  install(
    EXPORT HPXJSONTarget
    NAMESPACE JSON::
    FILE HPXJSONTarget.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
    COMPONENT cmake
  )

  add_library(JSON::json ALIAS json)
endif()
