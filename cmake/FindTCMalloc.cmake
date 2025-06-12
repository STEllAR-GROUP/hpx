# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_Tcmalloc_MINIMAL QUIET libtcmalloc_minimal)
pkg_check_modules(PC_Tcmalloc QUIET libtcmalloc)

find_path(
  Tcmalloc_INCLUDE_DIR google/tcmalloc.h
  HINTS ${TCMALLOC_ROOT}
        ENV
        TCMALLOC_ROOT
        ${HPX_TCMALLOC_ROOT}
        ${PC_Tcmalloc_MINIMAL_INCLUDEDIR}
        ${PC_Tcmalloc_MINIMAL_INCLUDE_DIRS}
        ${PC_Tcmalloc_INCLUDEDIR}
        ${PC_Tcmalloc_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  Tcmalloc_LIBRARY
  NAMES tcmalloc_minimal libtcmalloc_minimal tcmalloc libtcmalloc
  HINTS ${TCMALLOC_ROOT}
        ENV
        TCMALLOC_ROOT
        ${HPX_TCMALLOC_ROOT}
        ${PC_Tcmalloc_MINIMAL_LIBDIR}
        ${PC_Tcmalloc_MINIMAL_LIBRARY_DIRS}
        ${PC_Tcmalloc_LIBDIR}
        ${PC_Tcmalloc_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set Tcmalloc_ROOT
get_filename_component(Tcmalloc_ROOT ${Tcmalloc_INCLUDE_DIR} DIRECTORY)

set(Tcmalloc_LIBRARIES ${Tcmalloc_LIBRARY})
set(Tcmalloc_INCLUDE_DIRS ${Tcmalloc_INCLUDE_DIR})

find_package_handle_standard_args(
  TCMalloc DEFAULT_MSG Tcmalloc_LIBRARY Tcmalloc_INCLUDE_DIR
)

get_property(
  _type
  CACHE Tcmalloc_ROOT
  PROPERTY TYPE
)
if(_type)
  set_property(CACHE Tcmalloc_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE Tcmalloc_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(Tcmalloc_ROOT Tcmalloc_LIBRARY Tcmalloc_INCLUDE_DIR)
