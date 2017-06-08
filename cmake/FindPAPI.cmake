# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_PAPI QUIET papi)

find_path(PAPI_INCLUDE_DIR papi.h
  HINTS
    ${PAPI_ROOT} ENV PAPI_ROOT
    ${PC_PAPI_INCLUDEDIR}
    ${PC_PAPI_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(PAPI_LIBRARY NAMES papi libpapi
  HINTS
    ${PAPI_ROOT} ENV PAPI_ROOT
    ${PC_PAPI_LIBDIR}
    ${PC_PAPI_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(PAPI_LIBRARIES ${PAPI_LIBRARY})
set(PAPI_INCLUDE_DIRS ${PAPI_INCLUDE_DIR})

find_package_handle_standard_args(PAPI DEFAULT_MSG
  PAPI_LIBRARY PAPI_INCLUDE_DIR)

get_property(_type CACHE PAPI_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE PAPI_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE PAPI_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(PAPI_ROOT PAPI_LIBRARY PAPI_INCLUDE_DIR)
