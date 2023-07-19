# Copyright (c) 2017      Abhimanyu Rawat
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# compatibility with older CMake versions
if(LIBSIGSEGV_ROOT AND NOT Libsigsegv_ROOT)
  set(Libsigsegv_ROOT
      ${LIBSIGSEGV_ROOT}
      CACHE PATH "Libsigsegv base directory"
  )
  unset(LIBSIGSEGV_ROOT CACHE)
endif()

find_package(PkgConfig QUIET)
pkg_check_modules(PC_LIBSIGSEGV QUIET libsigsegv)

find_path(
  Libsigsegv_INCLUDE_DIR sigsegv.h
  HINTS ${Libsigsegv_ROOT}
        ENV
        Libsigsegv_ROOT
        ${PC_Libsigsegv_MINIMAL_INCLUDEDIR}
        ${PC_Libsigsegv_MINIMAL_INCLUDE_DIRS}
        ${PC_Libsigsegv_INCLUDEDIR}
        ${PC_Libsigsegv_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  Libsigsegv_LIBRARY
  NAMES sigsegv libsigsegv
  HINTS ${Libsigsegv_ROOT}
        ENV
        Libsigsegv_ROOT
        ${PC_Libsigsegv_MINIMAL_LIBDIR}
        ${PC_Libsigsegv_MINIMAL_LIBRARY_DIRS}
        ${PC_Libsigsegv_LIBDIR}
        ${PC_Libsigsegv_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set Libsigsegv_ROOT in case the other hints are used
if(Libsigsegv_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${Libsigsegv_ROOT} Libsigsegv_ROOT)
elseif("$ENV{LIBSIGSEGV_ROOT}")
  file(TO_CMAKE_PATH $ENV{LIBSIGSEGV_ROOT} Libsigsegv_ROOT)
else()
  file(TO_CMAKE_PATH "${Libsigsegv_INCLUDE_DIR}" Libsigsegv_INCLUDE_DIR)
  string(REPLACE "/include" "" Libsigsegv_ROOT "${Libsigsegv_INCLUDE_DIR}")
endif()

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
