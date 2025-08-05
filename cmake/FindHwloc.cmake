# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2023 Hartmut Kaiser
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
    Hwloc_INCLUDE_DIR hwloc.h
    HINTS ${HWLOC_ROOT}
          ENV
          HWLOC_ROOT
          ${HPX_HWLOC_ROOT}
          ${PC_Hwloc_MINIMAL_INCLUDEDIR}
          ${PC_Hwloc_MINIMAL_INCLUDE_DIRS}
          ${PC_Hwloc_INCLUDEDIR}
          ${PC_Hwloc_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )

  find_library(
    Hwloc_LIBRARY
    NAMES libhwloc.so libhwloc.lib hwloc
    HINTS ${HWLOC_ROOT}
          ENV
          HWLOC_ROOT
          ${HPX_Hwloc_ROOT}
          ${PC_Hwloc_MINIMAL_LIBDIR}
          ${PC_Hwloc_MINIMAL_LIBRARY_DIRS}
          ${PC_Hwloc_LIBDIR}
          ${PC_Hwloc_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
  )
  # Set Hwloc_ROOT
  get_filename_component(Hwloc_ROOT ${Hwloc_INCLUDE_DIR} DIRECTORY)

  set(Hwloc_LIBRARIES ${Hwloc_LIBRARY})
  set(Hwloc_INCLUDE_DIRS ${Hwloc_INCLUDE_DIR})

  find_package_handle_standard_args(
    Hwloc DEFAULT_MSG Hwloc_LIBRARY Hwloc_INCLUDE_DIR
  )

  get_property(
    _type
    CACHE Hwloc_ROOT
    PROPERTY TYPE
  )
  if(_type)
    set_property(CACHE Hwloc_ROOT PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE Hwloc_ROOT PROPERTY TYPE PATH)
    endif()
  endif()

  add_library(Hwloc::hwloc INTERFACE IMPORTED)
  target_include_directories(Hwloc::hwloc SYSTEM INTERFACE ${Hwloc_INCLUDE_DIR})
  target_link_libraries(Hwloc::hwloc INTERFACE ${Hwloc_LIBRARIES})

  mark_as_advanced(Hwloc_LIBRARY Hwloc_INCLUDE_DIR)
endif()
