
# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_VALGRIND QUIET valgrind)

find_path(VALGRIND_INCLUDE_DIR valgrind/valgrind.h
  HINTS
    ${VALGRIND_ROOT} ENV VALGRIND_ROOT
    ${PC_VALGRIND_INCLUDEDIR}
    ${PC_VALGRIND_INCLUDE_DIRS}
  PATH_SUFFIXES include)

set(VALGRIND_INCLUDE_DIRS ${VALGRIND_INCLUDE_DIR})

find_package_handle_standard_args(Valgrind DEFAULT_MSG
  VALGRIND_INCLUDE_DIR)

get_property(_type CACHE VALGRIND_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE VALGRIND_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE VALGRIND_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(VALGRIND_ROOT VALGRIND_INCLUDE_DIR)
