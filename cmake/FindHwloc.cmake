# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET Hwloc::hwloc)
  find_package(PkgConfig QUIET)
  pkg_check_modules(PC_HWLOC QUIET hwloc)

  find_path(
    HWLOC_INCLUDE_DIR hwloc.h
    HINTS ${HWLOC_ROOT}
          ENV
          HWLOC_ROOT
          ${HPX_HWLOC_ROOT}
          ${PC_HWLOC_MINIMAL_INCLUDEDIR}
          ${PC_HWLOC_MINIMAL_INCLUDE_DIRS}
          ${PC_HWLOC_INCLUDEDIR}
          ${PC_HWLOC_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )

  find_library(
    HWLOC_LIBRARY
    NAMES hwloc libhwloc
    HINTS ${HWLOC_ROOT}
          ENV
          HWLOC_ROOT
          ${HPX_HWLOC_ROOT}
          ${PC_HWLOC_MINIMAL_LIBDIR}
          ${PC_HWLOC_MINIMAL_LIBRARY_DIRS}
          ${PC_HWLOC_LIBDIR}
          ${PC_HWLOC_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
  )

  # Set HWLOC_ROOT in case the other hints are used
  if(HWLOC_ROOT)
    # The call to file is for compatibility with windows paths
    file(TO_CMAKE_PATH ${HWLOC_ROOT} HWLOC_ROOT)
  elseif("$ENV{HWLOC_ROOT}")
    file(TO_CMAKE_PATH $ENV{HWLOC_ROOT} HWLOC_ROOT)
  else()
    file(TO_CMAKE_PATH "${HWLOC_INCLUDE_DIR}" HWLOC_INCLUDE_DIR)
    string(REPLACE "/include" "" HWLOC_ROOT "${HWLOC_INCLUDE_DIR}")
  endif()

  set(HWLOC_LIBRARIES ${HWLOC_LIBRARY})
  set(HWLOC_INCLUDE_DIRS ${HWLOC_INCLUDE_DIR})

  find_package_handle_standard_args(
    Hwloc DEFAULT_MSG HWLOC_LIBRARY HWLOC_INCLUDE_DIR
  )

  get_property(
    _type
    CACHE HWLOC_ROOT
    PROPERTY TYPE
  )
  if(_type)
    set_property(CACHE HWLOC_ROOT PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE HWLOC_ROOT PROPERTY TYPE PATH)
    endif()
  endif()

  add_library(Hwloc::hwloc INTERFACE IMPORTED)
  target_include_directories(Hwloc::hwloc SYSTEM INTERFACE ${HWLOC_INCLUDE_DIR})
  target_link_libraries(Hwloc::hwloc INTERFACE ${HWLOC_LIBRARIES})

  mark_as_advanced(HWLOC_ROOT HWLOC_LIBRARY HWLOC_INCLUDE_DIR)
endif()
