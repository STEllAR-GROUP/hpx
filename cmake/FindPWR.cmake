# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2022 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET PWR::pwr)
  # compatibility with older CMake versions
  if(PWR_ROOT AND NOT Pwr_ROOT)
    set(Pwr_ROOT
        ${PWR_ROOT}
        CACHE PATH "PWR base directory"
    )
    unset(PWR_ROOT CACHE)
  endif()

  find_package(PkgConfig QUIET)
  pkg_check_modules(PC_Pwr QUIET pwr)
  if(NOT PC_Pwr_FOUND)
    pkg_check_modules(PC_Pwr pwrapi QUIET)
  endif()

  find_path(
    Pwr_INCLUDE_DIR pwr.h
    HINTS ${Pwr_ROOT}
          ENV
          PWR_ROOT
          ${HPX_Pwr_ROOT}
          ${PC_Pwr_MINIMAL_INCLUDEDIR}
          ${PC_Pwr_MINIMAL_INCLUDE_DIRS}
          ${PC_Pwr_INCLUDEDIR}
          ${PC_Pwr_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )

  find_library(
    Pwr_LIBRARY
    NAMES pwr libpwr
    HINTS ${Pwr_ROOT}
          ENV
          PWR_ROOT
          ${HPX_Pwr_ROOT}
          ${PC_Pwr_MINIMAL_LIBDIR}
          ${PC_Pwr_MINIMAL_LIBRARY_DIRS}
          ${PC_Pwr_LIBDIR}
          ${PC_Pwr_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
  )

  # Set Pwr_ROOT in case the other hints are used
  if(Pwr_ROOT)
    # The call to file is for compatibility with windows paths
    file(TO_CMAKE_PATH ${Pwr_ROOT} Pwr_ROOT)
  elseif(DEFINED ENV{PWR_ROOT})
    file(TO_CMAKE_PATH $ENV{PWR_ROOT} Pwr_ROOT)
  else()
    file(TO_CMAKE_PATH "${Pwr_INCLUDE_DIR}" Pwr_INCLUDE_DIR)
    string(REPLACE "/include" "" Pwr_ROOT "${Pwr_INCLUDE_DIR}")
  endif()

  set(Pwr_LIBRARIES ${Pwr_LIBRARY})
  set(Pwr_INCLUDE_DIRS ${Pwr_INCLUDE_DIR})

  find_package_handle_standard_args(PWR DEFAULT_MSG Pwr_LIBRARY Pwr_INCLUDE_DIR)

  get_property(
    _type
    CACHE Pwr_ROOT
    PROPERTY TYPE
  )
  if(_type)
    set_property(CACHE Pwr_ROOT PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE Pwr_ROOT PROPERTY TYPE PATH)
    endif()
  endif()

  add_library(PWR::pwr INTERFACE IMPORTED)
  target_include_directories(PWR::pwr SYSTEM INTERFACE ${Pwr_INCLUDE_DIR})
  target_link_libraries(PWR::pwr INTERFACE ${Pwr_LIBRARIES})

  mark_as_advanced(Pwr_ROOT Pwr_LIBRARY Pwr_INCLUDE_DIR)
endif()
