# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2023 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET Papi::papi)

  find_package(PkgConfig QUIET)
  pkg_check_modules(PC_Papi QUIET papi)

  find_path(
    Papi_INCLUDE_DIR papi.h
    HINTS ${PAPI_ROOT} ENV PAPI_ROOT ${HPX_PAPI_ROOT} ${PC_Papi_INCLUDEDIR}
          ${PC_Papi_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )

  find_library(
    Papi_LIBRARY
    NAMES papi libpapi
    HINTS ${PAPI_ROOT} ENV PAPI_ROOT ${HPX_PAPI_ROOT} ${PC_Papi_LIBDIR}
          ${PC_Papi_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
  )

  set(Papi_LIBRARIES ${Papi_LIBRARY})
  set(Papi_INCLUDE_DIRS ${Papi_INCLUDE_DIR})

  find_package_handle_standard_args(
    Papi REQUIRED_VARS Papi_LIBRARY Papi_INCLUDE_DIR
  )

  get_property(
    _type
    CACHE Papi_ROOT
    PROPERTY TYPE
  )
  if(_type)
    set_property(CACHE Papi_ROOT PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE Papi_ROOT PROPERTY TYPE PATH)
    endif()
  endif()

  mark_as_advanced(Papi_ROOT Papi_LIBRARY Papi_INCLUDE_DIR)

  add_library(Papi::papi INTERFACE IMPORTED)
  target_include_directories(Papi::papi SYSTEM INTERFACE ${Papi_INCLUDE_DIR})
  target_link_libraries(Papi::papi INTERFACE ${Papi_LIBRARY})
endif()
