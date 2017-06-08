# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_SNAPPY QUIET snappy)

find_path(SNAPPY_INCLUDE_DIR snappy.h
  HINTS
    ${SNAPPY_ROOT} ENV SNAPPY_ROOT
    ${PC_SNAPPY_MINIMAL_INCLUDEDIR}
    ${PC_SNAPPY_MINIMAL_INCLUDE_DIRS}
    ${PC_SNAPPY_INCLUDEDIR}
    ${PC_SNAPPY_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(SNAPPY_LIBRARY NAMES snappy libsnappy
  HINTS
    ${SNAPPY_ROOT} ENV SNAPPY_ROOT
    ${PC_SNAPPY_MINIMAL_LIBDIR}
    ${PC_SNAPPY_MINIMAL_LIBRARY_DIRS}
    ${PC_SNAPPY_LIBDIR}
    ${PC_SNAPPY_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(SNAPPY_LIBRARIES ${SNAPPY_LIBRARY})
set(SNAPPY_INCLUDE_DIRS ${SNAPPY_INCLUDE_DIR})

find_package_handle_standard_args(Snappy DEFAULT_MSG
  SNAPPY_LIBRARY SNAPPY_INCLUDE_DIR)

get_property(_type CACHE SNAPPY_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE SNAPPY_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE SNAPPY_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(SNAPPY_ROOT SNAPPY_LIBRARY SNAPPY_INCLUDE_DIR)
