# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2022 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET PWR::pwr)
  find_package(PkgConfig QUIET)
  pkg_check_modules(PC_PWR QUIET pwr)
  if(NOT PC_PWR_FOUND)
    pkg_check_modules(PC_PWR pwrapi QUIET)
  endif()

  find_path(
    PWR_INCLUDE_DIR pwr.h
    HINTS ${PWR_ROOT}
          ENV
          PWR_ROOT
          ${HPX_PWR_ROOT}
          ${PC_PWR_MINIMAL_INCLUDEDIR}
          ${PC_PWR_MINIMAL_INCLUDE_DIRS}
          ${PC_PWR_INCLUDEDIR}
          ${PC_PWR_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )

  find_library(
    PWR_LIBRARY
    NAMES pwr libpwr
    HINTS ${PWR_ROOT}
          ENV
          PWR_ROOT
          ${HPX_PWR_ROOT}
          ${PC_PWR_MINIMAL_LIBDIR}
          ${PC_PWR_MINIMAL_LIBRARY_DIRS}
          ${PC_PWR_LIBDIR}
          ${PC_PWR_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
  )

  # Set PWR_ROOT in case the other hints are used
  if(PWR_ROOT)
    # The call to file is for compatibility with windows paths
    file(TO_CMAKE_PATH ${PWR_ROOT} PWR_ROOT)
  elseif("$ENV{PWR_ROOT}")
    file(TO_CMAKE_PATH $ENV{PWR_ROOT} PWR_ROOT)
  else()
    file(TO_CMAKE_PATH "${PWR_INCLUDE_DIR}" PWR_INCLUDE_DIR)
    string(REPLACE "/include" "" PWR_ROOT "${PWR_INCLUDE_DIR}")
  endif()

  set(PWR_LIBRARIES ${PWR_LIBRARY})
  set(PWR_INCLUDE_DIRS ${PWR_INCLUDE_DIR})

  find_package_handle_standard_args(PWR DEFAULT_MSG PWR_LIBRARY PWR_INCLUDE_DIR)

  get_property(
    _type
    CACHE PWR_ROOT
    PROPERTY TYPE
  )
  if(_type)
    set_property(CACHE PWR_ROOT PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE PWR_ROOT PROPERTY TYPE PATH)
    endif()
  endif()

  add_library(PWR::pwr INTERFACE IMPORTED)
  target_include_directories(PWR::pwr SYSTEM INTERFACE ${PWR_INCLUDE_DIR})
  target_link_libraries(PWR::pwr INTERFACE ${PWR_LIBRARIES})

  mark_as_advanced(PWR_ROOT PWR_LIBRARY PWR_INCLUDE_DIR)
endif()
