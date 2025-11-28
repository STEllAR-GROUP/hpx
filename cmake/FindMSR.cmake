# Copyright (c)      2015 University of Oregon
# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_MSR QUIET libmsr)

find_path(
  Msr_INCLUDE_DIR msr_core.h
  HINTS ${MSR_ROOT} ENV MSR_ROOT ${PC_Msr_INCLUDEDIR} ${PC_Msr_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  Msr_LIBRARY
  NAMES msr libmsr
  HINTS ${MSR_ROOT} ENV MSR_ROOT ${PC_Msr_LIBDIR} ${PC_Msr_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

get_filename_component(Msr_ROOT ${Msr_INCLUDE_DIR} DIRECTORY)

set(Msr_LIBRARIES ${Msr_LIBRARY})
set(Msr_INCLUDE_DIRS ${Msr_INCLUDE_DIR})

find_package_handle_standard_args(MSR DEFAULT_MSG Msr_LIBRARY Msr_INCLUDE_DIR)

mark_as_advanced(Msr_ROOT Msr_LIBRARY Msr_INCLUDE_DIR)
