# Copyright (c) 2017      Abhimanyu Rawat
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_LIBSIGSEGV QUIET libsigsegv)

find_path(
  Libsigsegv_INCLUDE_DIR sigsegv.h
  HINTS ${LIBSIGSEGV_ROOT}
        ENV
        LIBSIGSEGV_ROOT
        ${PC_Libsigsegv_MINIMAL_INCLUDEDIR}
        ${PC_Libsigsegv_MINIMAL_INCLUDE_DIRS}
        ${PC_Libsigsegv_INCLUDEDIR}
        ${PC_Libsigsegv_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  Libsigsegv_LIBRARY
  NAMES sigsegv libsigsegv
  HINTS ${LIBSIGSEGV_ROOT}
        ENV
        LIBSIGSEGV_ROOT
        ${PC_Libsigsegv_MINIMAL_LIBDIR}
        ${PC_Libsigsegv_MINIMAL_LIBRARY_DIRS}
        ${PC_Libsigsegv_LIBDIR}
        ${PC_Libsigsegv_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

set(Libsigsegv_LIBRARIES ${Libsigsegv_LIBRARY})
set(Libsigsegv_INCLUDE_DIRS ${Libsigsegv_INCLUDE_DIR})

find_package_handle_standard_args(
  LibSigSegv DEFAULT_MSG Libsigsegv_LIBRARY Libsigsegv_INCLUDE_DIR
)

get_property(
  _type
  CACHE Libsigsegv_ROOT
  PROPERTY TYPE
)
if(_type)
  set_property(CACHE Libsigsegv_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE Libsigsegv_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(Libsigsegv_ROOT Libsigsegv_LIBRARY Libsigsegv_INCLUDE_DIR)
