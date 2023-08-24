# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2023 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# compatibility with older CMake versions
if(JEMALLOC_ROOT AND NOT Jemalloc_ROOT)
  set(Jemalloc_ROOT
      ${JEMALLOC_ROOT}
      CACHE PATH "Jemalloc base directory"
  )
  unset(JEMALLOC_ROOT CACHE)
endif()

find_package(PkgConfig QUIET)
pkg_check_modules(PC_JEMALLOC QUIET jemalloc)

find_path(
  Jemalloc_INCLUDE_DIR jemalloc/jemalloc.h
  HINTS ${Jemalloc_ROOT}
        ENV
        JEMALLOC_ROOT
        ${HPX_JEMALLOC_ROOT}
        ${PC_Jemalloc_MINIMAL_INCLUDEDIR}
        ${PC_Jemalloc_MINIMAL_INCLUDE_DIRS}
        ${PC_Jemalloc_INCLUDEDIR}
        ${PC_Jemalloc_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

if(MSVC)
  # MSVC needs additional header files provided by jemalloc to compensate for
  # missing posix headers
  find_path(
    Jemalloc_ADDITIONAL_INCLUDE_DIR msvc_compat/strings.h
    HINTS ${Jemalloc_ROOT}
          ENV
          JEMALLOC_ROOT
          ${HPX_JEMALLOC_ROOT}
          ${PC_Jemalloc_MINIMAL_INCLUDEDIR}
          ${PC_Jemalloc_MINIMAL_INCLUDE_DIRS}
          ${PC_Jemalloc_INCLUDEDIR}
          ${PC_Jemalloc_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )
endif()

# Set Jemalloc_ROOT in case the other hints are used
if(Jemalloc_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${Jemalloc_ROOT} Jemalloc_ROOT)
elseif("$ENV{JEMALLOC_ROOT}")
  file(TO_CMAKE_PATH $ENV{JEMALLOC_ROOT} Jemalloc_ROOT)
else()
  file(TO_CMAKE_PATH "${Jemalloc_INCLUDE_DIR}" Jemalloc_INCLUDE_DIR)
  string(REPLACE "/include" "" Jemalloc_ROOT "${Jemalloc_INCLUDE_DIR}")
endif()

find_library(
  Jemalloc_LIBRARY
  NAMES jemalloc libjemalloc
  HINTS ${Jemalloc_ROOT}
        ENV
        JEMALLOC_ROOT
        ${HPX_JEMALLOC_ROOT}
        ${PC_Jemalloc_MINIMAL_LIBDIR}
        ${PC_Jemalloc_MINIMAL_LIBRARY_DIRS}
        ${PC_Jemalloc_LIBDIR}
        ${PC_Jemalloc_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set Jemalloc_ROOT in case the other hints are used
if(NOT Jemalloc_ROOT AND "$ENV{JEMALLOC_ROOT}")
  set(Jemalloc_ROOT $ENV{JEMALLOC_ROOT})
elseif(NOT Jemalloc_ROOT)
  string(REPLACE "/include" "" Jemalloc_ROOT "${Jemalloc_INCLUDE_DIR}")
endif()

set(Jemalloc_LIBRARIES ${Jemalloc_LIBRARY})
set(Jemalloc_INCLUDE_DIRS ${Jemalloc_INCLUDE_DIR})

find_package_handle_standard_args(
  Jemalloc DEFAULT_MSG Jemalloc_LIBRARY Jemalloc_INCLUDE_DIR
)

get_property(
  _type
  CACHE Jemalloc_ROOT
  PROPERTY TYPE
)
if(_type)
  set_property(CACHE Jemalloc_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE Jemalloc_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(Jemalloc_ROOT Jemalloc_LIBRARY Jemalloc_INCLUDE_DIR)
