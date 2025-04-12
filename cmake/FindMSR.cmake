# Copyright (c)      2015 University of Oregon
# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# compatibility with older CMake versions
if(MSR_ROOT AND NOT Msr_ROOT)
  set(Msr_ROOT
      ${MSR_ROOT}
      CACHE PATH "Msr base directory"
  )
  unset(MSR_ROOT CACHE)
endif()

find_package(PkgConfig QUIET)
pkg_check_modules(PC_MSR QUIET libmsr)

find_path(
  Msr_INCLUDE_DIR msr_core.h
  HINTS ${Msr_ROOT} ENV MSR_ROOT ${PC_Msr_INCLUDEDIR} ${PC_Msr_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  Msr_LIBRARY
  NAMES msr libmsr
  HINTS ${Msr_ROOT} ENV MSR_ROOT ${PC_Msr_LIBDIR} ${PC_Msr_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set Msr_ROOT in case the other hints are used
if(Msr_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${Msr_ROOT} Msr_ROOT)
elseif(DEFINED ENV{MSR_ROOT})
  file(TO_CMAKE_PATH $ENV{MSR_ROOT} Msr_ROOT)
else()
  file(TO_CMAKE_PATH "${Msr_INCLUDE_DIR}" Msr_INCLUDE_DIR)
  string(REPLACE "/include" "" Msr_ROOT "${Msr_INCLUDE_DIR}")
endif()

set(Msr_LIBRARIES ${Msr_LIBRARY})
set(Msr_INCLUDE_DIRS ${Msr_INCLUDE_DIR})

find_package_handle_standard_args(MSR DEFAULT_MSG Msr_LIBRARY Msr_INCLUDE_DIR)

get_property(
  _type
  CACHE Msr_ROOT
  PROPERTY TYPE
)
if(_type)
  set_property(CACHE Msr_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE Msr_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(Msr_ROOT Msr_LIBRARY Msr_INCLUDE_DIR)
