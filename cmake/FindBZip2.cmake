# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig)
pkg_check_modules(PC_BZIP2 QUIET bzip2)

find_path(BZIP2_INCLUDE_DIR bzlib.h
  HINTS
    ${BZIP2_ROOT} ENV BZIP2_ROOT
    ${PC_BZIP2_MINIMAL_INCLUDEDIR}
    ${PC_BZIP2_MINIMAL_INCLUDE_DIRS}
    ${PC_BZIP2_INCLUDEDIR}
    ${PC_BZIP2_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(BZIP2_LIBRARY NAMES bzip2 libbz2 bz2
  HINTS
    ${BZIP2_ROOT} ENV BZIP2_ROOT
    ${PC_BZIP2_MINIMAL_LIBDIR}
    ${PC_BZIP2_MINIMAL_LIBRARY_DIRS}
    ${PC_BZIP2_LIBDIR}
    ${PC_BZIP2_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(BZIP2_LIBRARIES ${BZIP2_LIBRARY})
set(BZIP2_INCLUDE_DIRS ${BZIP2_INCLUDE_DIR})

find_package_handle_standard_args(BZip2 DEFAULT_MSG
  BZIP2_LIBRARY BZIP2_INCLUDE_DIR)

get_property(_type CACHE BZIP2_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE BZIP2_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE BZIP2_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(BZIP2_ROOT BZIP2_LIBRARY BZIP2_INCLUDE_DIR)
