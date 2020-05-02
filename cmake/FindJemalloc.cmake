# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_JEMALLOC QUIET jemalloc)

find_path(
  JEMALLOC_INCLUDE_DIR jemalloc/jemalloc.h
  HINTS ${JEMALLOC_ROOT}
        ENV
        JEMALLOC_ROOT
        ${HPX_JEMALLOC_ROOT}
        ${PC_JEMALLOC_MINIMAL_INCLUDEDIR}
        ${PC_JEMALLOC_MINIMAL_INCLUDE_DIRS}
        ${PC_JEMALLOC_INCLUDEDIR}
        ${PC_JEMALLOC_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

if(MSVC)
  # MSVC needs additional header files provided by jemalloc to compensate for
  # missing posix headers
  find_path(
    JEMALLOC_ADDITIONAL_INCLUDE_DIR msvc_compat/strings.h
    HINTS ${JEMALLOC_ROOT}
          ENV
          JEMALLOC_ROOT
          ${HPX_JEMALLOC_ROOT}
          ${PC_JEMALLOC_MINIMAL_INCLUDEDIR}
          ${PC_JEMALLOC_MINIMAL_INCLUDE_DIRS}
          ${PC_JEMALLOC_INCLUDEDIR}
          ${PC_JEMALLOC_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )
endif()

# Set JEMALLOC_ROOT in case the other hints are used
if(JEMALLOC_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${JEMALLOC_ROOT} JEMALLOC_ROOT)
elseif("$ENV{JEMALLOC_ROOT}")
  file(TO_CMAKE_PATH $ENV{JEMALLOC_ROOT} JEMALLOC_ROOT)
else()
  file(TO_CMAKE_PATH "${JEMALLOC_INCLUDE_DIR}" JEMALLOC_INCLUDE_DIR)
  string(REPLACE "/include" "" JEMALLOC_ROOT "${JEMALLOC_INCLUDE_DIR}")
endif()

find_library(
  JEMALLOC_LIBRARY
  NAMES jemalloc libjemalloc
  HINTS ${JEMALLOC_ROOT}
        ENV
        JEMALLOC_ROOT
        ${HPX_JEMALLOC_ROOT}
        ${PC_JEMALLOC_MINIMAL_LIBDIR}
        ${PC_JEMALLOC_MINIMAL_LIBRARY_DIRS}
        ${PC_JEMALLOC_LIBDIR}
        ${PC_JEMALLOC_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set JEMALLOC_ROOT in case the other hints are used
if(NOT JEMALLOC_ROOT AND "$ENV{JEMALLOC_ROOT}")
  set(JEMALLOC_ROOT $ENV{JEMALLOC_ROOT})
elseif(NOT JEMALLOC_ROOT)
  string(REPLACE "/include" "" JEMALLOC_ROOT "${JEMALLOC_INCLUDE_DIR}")
endif()

set(JEMALLOC_LIBRARIES ${JEMALLOC_LIBRARY})
set(JEMALLOC_INCLUDE_DIRS ${JEMALLOC_INCLUDE_DIR})

find_package_handle_standard_args(
  Jemalloc DEFAULT_MSG JEMALLOC_LIBRARY JEMALLOC_INCLUDE_DIR
)

get_property(
  _type
  CACHE JEMALLOC_ROOT
  PROPERTY TYPE
)
if(_type)
  set_property(CACHE JEMALLOC_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE JEMALLOC_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(JEMALLOC_ROOT JEMALLOC_LIBRARY JEMALLOC_INCLUDE_DIR)
