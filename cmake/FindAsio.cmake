# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET Asio::asio)

  find_path(
    Asio_INCLUDE_DIR asio.hpp
    HINTS "${ASIO_ROOT}" ENV ASIO_ROOT "${HPX_ASIO_ROOT}"
    PATH_SUFFIXES include
  )

  get_filename_component(Asio_ROOT ${Asio_INCLUDE_DIR} DIRECTORY)

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
  )

  add_library(Asio::asio INTERFACE IMPORTED)
  target_include_directories(Asio::asio SYSTEM INTERFACE ${Asio_INCLUDE_DIR})

  mark_as_advanced(Asio_ROOT Asio_INCLUDE_DIR Asio_VERSION_STRING)
endif()
