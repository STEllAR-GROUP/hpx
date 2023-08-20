# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# compatibility with older CMake versions
if(VALGRIND_ROOT AND NOT Valgrind_ROOT)
  set(Valgrind_ROOT
      ${VALGRIND_ROOT}
      CACHE PATH "Valgrind base directory"
  )
  unset(VALGRIND_ROOT CACHE)
endif()

find_package(PkgConfig QUIET)
pkg_check_modules(PC_VALGRIND QUIET valgrind)

find_path(
  Valgrind_INCLUDE_DIR valgrind/valgrind.h
  HINTS ${Valgrind_ROOT} ENV VALGRIND_ROOT ${PC_Valgrind_INCLUDEDIR}
        ${PC_Valgrind_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

set(Valgrind_INCLUDE_DIRS ${Valgrind_INCLUDE_DIR})

find_package_handle_standard_args(Valgrind DEFAULT_MSG Valgrind_INCLUDE_DIR)

get_property(
  _type
  CACHE Valgrind_ROOT
  PROPERTY TYPE
)
if(_type)
  set_property(CACHE Valgrind_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE Valgrind_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(Valgrind_ROOT Valgrind_INCLUDE_DIR)
