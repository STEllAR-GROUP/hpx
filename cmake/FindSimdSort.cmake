# Copyright (c)      2022 Srinivas Yadav
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET SimdSort::simdsort)
  find_path(
    SIMD_SORT_INCLUDE_DIR simd_sort/simd_sort.hpp
    HINTS "${SIMD_SORT_ROOT}" ENV SIMD_SORT_ROOT "${HPX_SIMD_SORT_ROOT}"
    PATH_SUFFIXES include
  )

  if(NOT SIMD_SORT_INCLUDE_DIR)
    hpx_error("Simd Sort not found")
  endif()

  # Set EVE_ROOT in case the other hints are used
  if(SIMD_SORT)
    # The call to file is for compatibility with windows paths
    file(TO_CMAKE_PATH ${SIMD_SORT} SIMD_SORT)
  elseif("$ENV{SIMD_SORT}")
    file(TO_CMAKE_PATH $ENV{SIMD_SORT} SIMD_SORT)
  else()
    file(TO_CMAKE_PATH "${SIMD_SORT_INCLUDE_DIR}" SIMD_SORT_INCLUDE_DIR)
    string(REPLACE "/include" "" SIMD_SORT_ROOT "${SIMD_SORT_INCLUDE_DIR}")
  endif()

  # if(SIMD_SORT_INCLUDE_DIR AND EXISTS
  # "${SIMD_SORT_INCLUDE_DIR}/eve/version.hpp") # Matches a line of the form: #
  # # #define EVE_VERSION "AA.BB.CC.DD" # # with arbitrary whitespace between
  # the tokens file( STRINGS "${SIMD_SORT_INCLUDE_DIR}/eve/version.hpp"
  # EVE_VERSION_DEFINE_LINE REGEX "#define[ \t]+EVE_LIB_VERSION[
  # \t]+\"+[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\"[ \t]*" ) # Extracts the dotted
  # version number in quotation marks as # SIMD_SORT_VERSION_STRING string(REGEX
  # REPLACE "#define EVE_LIB_VERSION \"([0-9]+\.[0-9]+\.[0-9]+\.[0-9])\"" "\\1"
  # SIMD_SORT_VERSION_STRING "${EVE_VERSION_DEFINE_LINE}" ) else() hpx_error(
  # "Could not find EVE_ROOT/include/eve/version.hpp. Please check your eve
  # installation" ) endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    SimdSort
    REQUIRED_VARS SIMD_SORT_INCLUDE_DIR
    VERSION_VAR SIMD_SORT_VERSION_STRING
  )

  add_library(SimdSort::simdsort INTERFACE IMPORTED)
  target_include_directories(
    SimdSort::simdsort SYSTEM INTERFACE ${SIMD_SORT_INCLUDE_DIR}
  )

  mark_as_advanced(
    SIMD_SORT_ROOT SIMD_SORT_INCLUDE_DIR SIMD_SORT_VERSION_STRING
  )
endif()
