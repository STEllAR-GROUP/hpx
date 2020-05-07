# Copyright (c) 2017      Abhimanyu Rawat
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_LIBSIGSEGV QUIET libsigsegv)

find_path(
  LIBSIGSEGV_INCLUDE_DIR sigsegv.h
  HINTS ${LIBSIGSEGV_ROOT}
        ENV
        LIBSIGSEGV_ROOT
        ${PC_LIBSIGSEGV_MINIMAL_INCLUDEDIR}
        ${PC_LIBSIGSEGV_MINIMAL_INCLUDE_DIRS}
        ${PC_LIBSIGSEGV_INCLUDEDIR}
        ${PC_LIBSIGSEGV_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  LIBSIGSEGV_LIBRARY
  NAMES sigsegv libsigsegv
  HINTS ${LIBSIGSEGV_ROOT}
        ENV
        LIBSIGSEGV_ROOT
        ${PC_LIBSIGSEGV_MINIMAL_LIBDIR}
        ${PC_LIBSIGSEGV_MINIMAL_LIBRARY_DIRS}
        ${PC_LIBSIGSEGV_LIBDIR}
        ${PC_LIBSIGSEGV_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set LIBSIGSEGV_ROOT in case the other hints are used
if(LIBSIGSEGV_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${LIBSIGSEGV_ROOT} LIBSIGSEGV_ROOT)
elseif("$ENV{LIBSIGSEGV_ROOT}")
  file(TO_CMAKE_PATH $ENV{LIBSIGSEGV_ROOT} LIBSIGSEGV_ROOT)
else()
  file(TO_CMAKE_PATH "${LIBSIGSEGV_INCLUDE_DIR}" LIBSIGSEGV_INCLUDE_DIR)
  string(REPLACE "/include" "" LIBSIGSEGV_ROOT "${LIBSIGSEGV_INCLUDE_DIR}")
endif()

set(LIBSIGSEGV_LIBRARIES ${LIBSIGSEGV_LIBRARY})
set(LIBSIGSEGV_INCLUDE_DIRS ${LIBSIGSEGV_INCLUDE_DIR})

find_package_handle_standard_args(
  LibSigSegv DEFAULT_MSG LIBSIGSEGV_LIBRARY LIBSIGSEGV_INCLUDE_DIR
)

get_property(
  _type
  CACHE LIBSIGSEGV_ROOT
  PROPERTY TYPE
)
if(_type)
  set_property(CACHE LIBSIGSEGV_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE LIBSIGSEGV_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(LIBSIGSEGV_ROOT LIBSIGSEGV_LIBRARY LIBSIGSEGV_INCLUDE_DIR)
