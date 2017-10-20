# Copyright (c)      2015 University of Oregon
# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_MSR QUIET libmsr)

find_path(MSR_INCLUDE_DIR msr_core.h
  HINTS
    ${MSR_ROOT} ENV MSR_ROOT
    ${PC_MSR_INCLUDEDIR}
    ${PC_MSR_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(MSR_LIBRARY NAMES msr libmsr
  HINTS
    ${MSR_ROOT} ENV MSR_ROOT
    ${PC_MSR_LIBDIR}
    ${PC_MSR_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(MSR_LIBRARIES ${MSR_LIBRARY})
set(MSR_INCLUDE_DIRS ${MSR_INCLUDE_DIR})

find_package_handle_standard_args(MSR DEFAULT_MSG
  MSR_LIBRARY MSR_INCLUDE_DIR)

get_property(_type CACHE MSR_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE MSR_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE MSR_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(MSR_ROOT MSR_LIBRARY MSR_INCLUDE_DIR)
