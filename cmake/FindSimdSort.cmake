# Copyright (c)      2023 Hari Hara Naveen S
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET SimdSort::simdsort)
  find_path(SIMD_SORT_INCLUDE_DIR avx512-16bit-qsort.hpp
            HINTS "${SIMD_SORT_ROOT}" ENV SIMD_SORT_ROOT
                  "${HPX_SIMD_SORT_ROOT}"
  )

  if(NOT SIMD_SORT_INCLUDE_DIR)
    hpx_error("Simd Sort not found")
  endif()

  if(SIMD_SORT)
    # The call to file is for compatibility with windows paths
    file(TO_CMAKE_PATH ${SIMD_SORT} SIMD_SORT)
  elseif("$ENV{SIMD_SORT}")
    file(TO_CMAKE_PATH $ENV{SIMD_SORT} SIMD_SORT)
  else()
    file(TO_CMAKE_PATH "${SIMD_SORT_INCLUDE_DIR}" SIMD_SORT_INCLUDE_DIR)
    string(REPLACE "/src" "" SIMD_SORT_ROOT "${SIMD_SORT_INCLUDE_DIR}")
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    SimdSort
    REQUIRED_VARS SIMD_SORT_INCLUDE_DIR
    VERSION_VAR SIMD_SORT_VERSION_STRING
  )

  add_library(SimdSort::simdsort PRIVATE IMPORTED)
  target_include_directories(
    SimdSort::simdsort SYSTEM PRIVATE ${SIMD_SORT_INCLUDE_DIR}
  )

  mark_as_advanced(
    SIMD_SORT_ROOT SIMD_SORT_INCLUDE_DIR SIMD_SORT_VERSION_STRING
  )
endif()
