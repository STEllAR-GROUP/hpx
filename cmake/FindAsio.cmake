# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET Asio::asio)
  # compatibility with older CMake versions
  if(ASIO_ROOT AND NOT Asio_ROOT)
    set(Asio_ROOT
        ${ASIO_ROOT}
        CACHE PATH "Asio base directory"
    )
    unset(ASIO_ROOT CACHE)
  endif()

  find_path(
    Asio_INCLUDE_DIR asio.hpp
    HINTS "${Asio_ROOT}" ENV ASIO_ROOT "${HPX_ASIO_ROOT}"
    PATH_SUFFIXES include
  )

  if(NOT Asio_INCLUDE_DIR)
    hpx_error(
      "Could not find Asio. Set Asio_ROOT as a CMake or environment variable to point to the Asio root install directory. Alternatively, set HPX_WITH_FETCH_ASIO=ON to fetch Asio using CMake's FetchContent (when using this option Asio will be installed together with HPX, be careful about conflicts with separately installed versions of Asio)."
    )
  endif()

  # Set Asio_ROOT in case the other hints are used
  if(Asio_ROOT)
    # The call to file is for compatibility with windows paths
    file(TO_CMAKE_PATH ${Asio_ROOT} Asio_ROOT)
  elseif("$ENV{ASIO_ROOT}")
    file(TO_CMAKE_PATH $ENV{ASIO_ROOT} Asio_ROOT)
  else()
    file(TO_CMAKE_PATH "${Asio_INCLUDE_DIR}" Asio_INCLUDE_DIR)
    string(REPLACE "/include" "" Asio_ROOT "${Asio_INCLUDE_DIR}")
  endif()

  if(Asio_INCLUDE_DIR AND EXISTS "${Asio_INCLUDE_DIR}/asio/version.hpp")
    # Matches a line of the form:
    #
    # #define ASIO_VERSION XXYYZZ // XX.YY.ZZ
    #
    # with arbitrary whitespace between the tokens
    file(
      STRINGS "${Asio_INCLUDE_DIR}/asio/version.hpp" Asio_VERSION_DEFINE_LINE
      REGEX
        "#define[ \t]+ASIO_VERSION[ \t]+[0-9]+[ \t]+//[ \t]+[0-9]+\.[0-9]+\.[0-9]+[ \t]*"
    )
    # Extracts the dotted version number after the comment as
    # Asio_VERSION_STRING
    string(REGEX
           REPLACE "#define ASIO_VERSION [0-9]+ // ([0-9]+\.[0-9]+\.[0-9]+)"
                   "\\1" Asio_VERSION_STRING "${Asio_VERSION_DEFINE_LINE}"
    )
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    Asio
    REQUIRED_VARS Asio_INCLUDE_DIR
    VERSION_VAR Asio_VERSION_STRING
    FOUND_VAR Asio_FOUND
  )

  add_library(Asio::asio INTERFACE IMPORTED)
  target_include_directories(Asio::asio SYSTEM INTERFACE ${Asio_INCLUDE_DIR})

  mark_as_advanced(Asio_ROOT Asio_INCLUDE_DIR Asio_VERSION_STRING)
endif()
