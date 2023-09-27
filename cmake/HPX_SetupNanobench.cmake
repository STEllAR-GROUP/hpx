# Copyright (c) 2022 Srinivas Yadav
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(("${HPX_WITH_BENCHMARK_BACKEND}" STREQUAL "NANOBENCH") AND NOT TARGET nanobench::nanobench)
  if(HPX_WITH_FETCH_NANOBENCH)
    if(FETCHCONTENT_SOURCE_DIR_NANOBENCH)
      hpx_info(
        "HPX_WITH_FETCH_NANOBENCH=${HPX_WITH_FETCH_NANOBENCH}, NANOBENCH will be used through CMake's FetchContent and installed alongside HPX (FETCHCONTENT_SOURCE_DIR_NANOBENCH=${FETCHCONTENT_SOURCE_DIR_NANOBENCH})"
      )
    else()
      hpx_info(
        "HPX_WITH_FETCH_NANOBENCH=${HPX_WITH_FETCH_NANOBENCH}, NANOBENCH will be fetched using CMake's FetchContent and installed alongside HPX (HPX_WITH_NANOBENCH_TAG=${HPX_WITH_NANOBENCH_TAG})"
      )
    endif()
    include(FetchContent)
    fetchcontent_declare(
      nanobench
      GIT_REPOSITORY https://github.com/martinus/nanobench.git
      GIT_TAG ${HPX_WITH_NANOBENCH_TAG}
    )

    fetchcontent_getproperties(nanobench)
    if(NOT nanobench_POPULATED)
      fetchcontent_populate(nanobench)
    endif()
    set(NANOBENCH_ROOT ${nanobench_SOURCE_DIR})

    add_library(nanobench INTERFACE)
    target_include_directories(
      nanobench SYSTEM INTERFACE $<BUILD_INTERFACE:${NANOBENCH_ROOT}/include/>
                                 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    install(
      TARGETS nanobench
      EXPORT HPXNanobenchTarget
      COMPONENT core
    )

    install(
      FILES ${NANOBENCH_ROOT}/include/nanobench.h
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      COMPONENT core
    )

    export(
      TARGETS nanobench
      NAMESPACE nanobench::
      FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXNanobenchTarget.cmake"
    )

    install(
      EXPORT HPXNanobenchTarget
      NAMESPACE nanobench::
      FILE HPXNanobenchTarget.cmake
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
      COMPONENT cmake
    )

    add_library(nanobench::nanobench ALIAS nanobench)

  else()
    if(NANOBENCH_ROOT)
      find_package(nanobench REQUIRED PATHS ${NANOBENCH_ROOT})
    else()
      hpx_error("NANOBENCH_ROOT not set")
    endif()
  endif()
endif()
