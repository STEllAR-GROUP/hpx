# Copyright (c)      2022 Srinivas Yadav
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET Eve::eve)
  find_path(
    EVE_INCLUDE_DIR eve/eve.hpp
    HINTS "${EVE_ROOT}" ENV EVE_ROOT "${HPX_EVE_ROOT}"
    PATH_SUFFIXES include
  )

  if(NOT EVE_INCLUDE_DIR)
    hpx_error(
      "Could not find Eve. Set EVE_ROOT as a CMake or environment variable to point to the Eve root install directory. Alternatively, set HPX_WITH_FETCH_EVE=ON to fetch Eve using CMake's FetchContent (when using this option Eve will be installed together with HPX, be careful about conflicts with separately installed versions of Eve)."
    )
  endif()

  # Set EVE_ROOT in case the other hints are used
  if(EVE_ROOT)
    # The call to file is for compatibility with windows paths
    file(TO_CMAKE_PATH ${EVE_ROOT} EVE_ROOT)
  elseif("$ENV{EVE_ROOT}")
    file(TO_CMAKE_PATH $ENV{EVE_ROOT} EVE_ROOT)
  else()
    file(TO_CMAKE_PATH "${EVE_INCLUDE_DIR}" EVE_INCLUDE_DIR)
    string(REPLACE "/include" "" EVE_ROOT "${EVE_INCLUDE_DIR}")
  endif()

  if(EVE_INCLUDE_DIR AND EXISTS "${EVE_INCLUDE_DIR}/eve/version.hpp")
    # Matches a line of the form:
    #
    # #define EVE_VERSION "AA.BB.CC.DD"
    #
    # with arbitrary whitespace between the tokens
    file(
      STRINGS "${EVE_INCLUDE_DIR}/eve/version.hpp" EVE_VERSION_DEFINE_LINE
      REGEX
        "#define[ \t]+EVE_LIB_VERSION[ \t]+\"+[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\"[ \t]*"
    )
    # Extracts the dotted version number in quotation marks as
    # EVE_VERSION_STRING
    string(REGEX
           REPLACE "#define EVE_LIB_VERSION \"([0-9]+\.[0-9]+\.[0-9]+\.[0-9])\""
                   "\\1" EVE_VERSION_STRING "${EVE_VERSION_DEFINE_LINE}"
    )
  else()
    set(EVE_VERSION_STRING "NOT_FOUND")
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    Eve
    REQUIRED_VARS EVE_INCLUDE_DIR
    VERSION_VAR EVE_VERSION_STRING
  )

  add_library(Eve::eve INTERFACE IMPORTED)
  target_include_directories(Eve::eve SYSTEM INTERFACE ${EVE_INCLUDE_DIR})

  mark_as_advanced(EVE_ROOT EVE_INCLUDE_DIR EVE_VERSION_STRING)
endif()
