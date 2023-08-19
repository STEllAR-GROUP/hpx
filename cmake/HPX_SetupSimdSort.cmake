# Copyright (c) 2022 Srinivas Yadav
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_FETCH_SIMD_SORT)
  if(FETCHCONTENT_SOURCE_DIR_SIMD_SORT)
    hpx_info(
      "HPX_WITH_FETCH_SIMD_SORT=${HPX_WITH_FETCH_SIMD_SORT}, x86-simd-sort will be used through CMake's FetchContent and installed alongside HPX (FETCHCONTENT_SOURCE_DIR_EVE=${FETCHCONTENT_SOURCE_DIR_EVE})"
    )
  else()
    hpx_info(
      "HPX_WITH_FETCH_SIMD_SORT=${HPX_WITH_FETCH_SIMD_SORT}, EVE will be fetched using CMake's FetchContent and installed alongside HPX (HPX_WITH_SIMD_SORT_TAG=${HPX_WITH_SIMD_SORT_TAG})"
    )
  endif()
  include(FetchContent)
  fetchcontent_declare(
    simdsort
    GIT_REPOSITORY https://github.com/intel/x86-simd-sort
    GIT_TAG ${HPX_WITH_SIMD_SORT_TAG}
  )

  fetchcontent_getproperties(simdsort)
  if(NOT simdsort_POPULATED)
    fetchcontent_populate(simdsort)
  endif()
  set(SIMD_SORT_ROOT ${simdsort_SOURCE_DIR})

  add_library(simdsort INTERFACE)
  target_include_directories(
    simdsort SYSTEM INTERFACE $<BUILD_INTERFACE:${SIMD_SORT_ROOT}/include/>
                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  install(
    TARGETS simdsort
    EXPORT HPXSimdSortTarget
    COMPONENT core
  )

  install(
    DIRECTORY ${SIMD_SORT_ROOT}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT core
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.ipp"
  )

  export(
    TARGETS simdsort
    NAMESPACE SimdSort::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXSimdSortTarget.cmake"
  )

  install(
    EXPORT HPXSimdSortTarget
    NAMESPACE SimdSort::
    FILE HPXSimdSortTarget.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
  )

  add_library(SimdSort::simdsort ALIAS simdsort)

endif()
