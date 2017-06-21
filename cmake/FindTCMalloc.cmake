
# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_TCMALLOC_MINIMAL QUIET libtcmalloc_minimal)
pkg_check_modules(PC_TCMALLOC QUIET libtcmalloc)

find_path(TCMALLOC_INCLUDE_DIR google/tcmalloc.h
  HINTS
    ${TCMALLOC_ROOT} ENV TCMALLOC_ROOT
    ${PC_TCMALLOC_MINIMAL_INCLUDEDIR}
    ${PC_TCMALLOC_MINIMAL_INCLUDE_DIRS}
    ${PC_TCMALLOC_INCLUDEDIR}
    ${PC_TCMALLOC_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(TCMALLOC_LIBRARY NAMES tcmalloc_minimal libtcmalloc_minimal tcmalloc libtcmalloc
  HINTS
    ${TCMALLOC_ROOT} ENV TCMALLOC_ROOT
    ${PC_TCMALLOC_MINIMAL_LIBDIR}
    ${PC_TCMALLOC_MINIMAL_LIBRARY_DIRS}
    ${PC_TCMALLOC_LIBDIR}
    ${PC_TCMALLOC_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(TCMALLOC_LIBRARIES ${TCMALLOC_LIBRARY})
set(TCMALLOC_INCLUDE_DIRS ${TCMALLOC_INCLUDE_DIR})

find_package_handle_standard_args(TCMalloc DEFAULT_MSG
  TCMALLOC_LIBRARY TCMALLOC_INCLUDE_DIR)

get_property(_type CACHE TCMALLOC_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE TCMALLOC_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE TCMALLOC_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(TCMALLOC_ROOT TCMALLOC_LIBRARY TCMALLOC_INCLUDE_DIR)
