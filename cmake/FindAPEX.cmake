# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig)
pkg_check_modules(PC_APEX QUIET APEX)

# This if statement is specific to APEX, and should not be copied into other
# Find cmake scripts.
if(NOT APEX_ROOT AND NOT $ENV{HOME_APEX} STREQUAL "")
  set(APEX_ROOT "$ENV{HOME_APEX}")
endif()

find_path(APEX_INCLUDE_DIR apex.hpp
  HINTS
    ${APEX_ROOT} ENV APEX_ROOT
    ${PC_APEX_INCLUDEDIR}
    ${PC_APEX_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(APEX_LIBRARY NAMES Apex libApex
  HINTS
    ${APEX_ROOT} ENV APEX_ROOT
    ${PC_APEX_LIBDIR}
    ${PC_APEX_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(APEX_LIBRARIES ${APEX_LIBRARY})
set(APEX_INCLUDE_DIRS ${APEX_INCLUDE_DIR})

find_package_handle_standard_args(APEX DEFAULT_MSG
  APEX_LIBRARY APEX_INCLUDE_DIR)

get_property(_type CACHE APEX_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE APEX_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE APEX_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(APEX_ROOT APEX_LIBRARY APEX_INCLUDE_DIR)
