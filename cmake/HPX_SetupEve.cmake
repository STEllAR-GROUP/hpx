# Copyright (c) 2022 Srinivas Yadav
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(("${HPX_WITH_DATAPAR_BACKEND}" STREQUAL "EVE") AND NOT TARGET eve::eve)
  if(HPX_WITH_FETCH_EVE)
    if(FETCHCONTENT_SOURCE_DIR_EVE)
      hpx_info(
        "HPX_WITH_FETCH_EVE=${HPX_WITH_FETCH_EVE}, EVE will be used through CMake's FetchContent and installed alongside HPX (FETCHCONTENT_SOURCE_DIR_EVE=${FETCHCONTENT_SOURCE_DIR_EVE})"
      )
    else()
      hpx_info(
        "HPX_WITH_FETCH_EVE=${HPX_WITH_FETCH_EVE}, EVE will be fetched using CMake's FetchContent and installed alongside HPX (HPX_WITH_EVE_TAG=${HPX_WITH_EVE_TAG})"
      )
    endif()
    include(FetchContent)
    fetchcontent_declare(
      eve
      GIT_REPOSITORY https://github.com/jfalcou/eve.git
      GIT_TAG ${HPX_WITH_EVE_TAG}
    )

    fetchcontent_getproperties(eve)
    if(NOT eve_POPULATED)
      fetchcontent_populate(eve)
    endif()
    set(EVE_ROOT ${eve_SOURCE_DIR})

    add_library(eve INTERFACE)
    target_include_directories(
      eve SYSTEM INTERFACE $<BUILD_INTERFACE:${EVE_ROOT}/include/>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    install(
      TARGETS eve
      EXPORT HPXEveTarget
      COMPONENT core
    )

    install(
      DIRECTORY ${EVE_ROOT}/include/
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      COMPONENT core
      FILES_MATCHING
      PATTERN "*.hpp"
      PATTERN "*.ipp"
    )

    export(
      TARGETS eve
      NAMESPACE eve::
      FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXEveTarget.cmake"
    )

    install(
      EXPORT HPXEveTarget
      NAMESPACE eve::
      FILE HPXEveTarget.cmake
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
    )

    add_library(eve::eve ALIAS eve)

  else()
    if(EVE_ROOT)
      find_package(eve REQUIRED PATHS ${EVE_ROOT})
    else()
      hpx_error("EVE_ROOT not set")
    endif()
  endif()
endif()
