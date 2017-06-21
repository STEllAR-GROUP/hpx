# - Try to find GooglePerftools
# Once done this will define
#  GOOGLE_PERFTOOLS_FOUND - System has GooglePerftools
#  GOOGLE_PERFTOOLS_INCLUDE_DIRS - The GooglePerftools include directory
#  GOOGLE_PERFTOOLS_LIBRARIES - Link these to use GooglePerftools

# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_GOOGLE_PERFTOOLS QUIET libprofiler)

find_path(GOOGLE_PERFTOOLS_INCLUDE_DIR google/profiler.h
  HINTS
    ${GOOGLE_PERFTOOLS_ROOT} ENV GOOGLE_PERFTOOLS_ROOT
    ${PC_GOOGLE_PERFTOOLS_INCLUDEDIR}
    ${PC_GOOGLE_PERFTOOLS_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(GOOGLE_PERFTOOLS_LIBRARY NAMES profiler libprofiler
  HINTS
    ${GOOGLE_PERFTOOLS_ROOT} ENV GOOGLE_PERFTOOLS_ROOT
    ${PC_GOOGLE_PERFTOOLS_LIBDIR}
    ${PC_GOOGLE_PERFTOOLS_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(GOOGLE_PERFTOOLS_LIBRARIES ${GOOGLE_PERFTOOLS_LIBRARY})
set(GOOGLE_PERFTOOLS_INCLUDE_DIRS ${GOOGLE_PERFTOOLS_INCLUDE_DIR})

find_package_handle_standard_args(GooglePerftools DEFAULT_MSG
  GOOGLE_PERFTOOLS_LIBRARY GOOGLE_PERFTOOLS_INCLUDE_DIR)

get_property(_type CACHE GOOGLE_PERFTOOLS_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE GOOGLE_PERFTOOLS_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE GOOGLE_PERFTOOLS_ROOT PROPERTY TYPE PATH)
  endif()
endif()

if(NOT GOOGLEPERFTOOLS_FOUND)
  hpx_info("    Set GOOGLE_PERFTOOLS_ROOT to the location where the gperftools are installed\n")
endif()

set(GOOGLE_PERFTOOLS_FOUND ${GOOGLEPERFTOOLS_FOUND})
unset(GOOGLEPERFTOOLS_FOUND)

mark_as_advanced(GOOGLE_PERFTOOLS_ROOT GOOGLE_PERFTOOLS_LIBRARY GOOGLE_PERFTOOLS_INCLUDE_DIR)
