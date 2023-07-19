# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2013-2023 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# compatibility with older CMake versions
if(SNAPPY_ROOT AND NOT Snappy_ROOT)
  set(Snappy_ROOT
      ${SNAPPY_ROOT}
      CACHE PATH "Snappy base directory"
  )
  unset(SNAPPY_ROOT CACHE)
endif()

find_package(PkgConfig QUIET)
pkg_check_modules(PC_SNAPPY QUIET snappy)

find_path(
  Snappy_INCLUDE_DIR snappy.h
  HINTS ${Snappy_ROOT}
        ENV
        SNAPPY_ROOT
        ${PC_Snappy_MINIMAL_INCLUDEDIR}
        ${PC_Snappy_MINIMAL_INCLUDE_DIRS}
        ${PC_Snappy_INCLUDEDIR}
        ${PC_Snappy_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  Snappy_LIBRARY
  NAMES snappy libsnappy
  HINTS ${Snappy_ROOT}
        ENV
        SNAPPY_ROOT
        ${PC_Snappy_MINIMAL_LIBDIR}
        ${PC_Snappy_MINIMAL_LIBRARY_DIRS}
        ${PC_Snappy_LIBDIR}
        ${PC_Snappy_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

set(Snappy_LIBRARIES ${Snappy_LIBRARY})
set(Snappy_INCLUDE_DIRS ${Snappy_INCLUDE_DIR})

find_package_handle_standard_args(
  Snappy DEFAULT_MSG Snappy_LIBRARY Snappy_INCLUDE_DIR
)

get_property(
  _type
  CACHE Snappy_ROOT
  PROPERTY TYPE
)
if(_type)
  set_property(CACHE Snappy_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE Snappy_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(Snappy_ROOT Snappy_LIBRARY Snappy_INCLUDE_DIR)
