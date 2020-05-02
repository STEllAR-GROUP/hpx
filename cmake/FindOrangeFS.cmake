# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2013      Shuangyang Yang
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_ORANGEFS QUIET orangefs)

find_path(
  ORANGEFS_INCLUDE_DIR
  NAMES pxfs.h orange.h
  HINTS ${ORANGEFS_ROOT} ENV ORANGEFS_ROOT ${PC_ORANGEFS_INCLUDEDIR}
        ${PC_ORANGEFS_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  ORANGEFS_LIBRARY
  NAMES pvfs2 # orangefs pvfs2 orangefsposix
  HINTS ${ORANGEFS_ROOT} ENV ORANGEFS_ROOT ${PC_ORANGEFS_LIBDIR}
        ${PC_ORANGEFS_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set ORANGEFS_ROOT in case the other hints are used
if(ORANGEFS_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${ORANGEFS_ROOT} ORANGEFS_ROOT)
elseif("$ENV{ORANGEFS_ROOT}")
  file(TO_CMAKE_PATH $ENV{ORANGEFS_ROOT} ORANGEFS_ROOT)
else()
  file(TO_CMAKE_PATH "${ORANGEFS_INCLUDE_DIR}" ORANGEFS_INCLUDE_DIR)
  string(REPLACE "/include" "" ORANGEFS_ROOT "${ORANGEFS_INCLUDE_DIR}")
endif()

set(ORANGEFS_LIBRARIES ${ORANGEFS_LIBRARY})
set(ORANGEFS_INCLUDE_DIRS ${ORANGEFS_INCLUDE_DIR})

find_package_handle_standard_args(
  OrangeFS DEFAULT_MSG ORANGEFS_LIBRARY ORANGEFS_INCLUDE_DIR
)

foreach(v ORANGEFS_ROOT)
  get_property(
    _type
    CACHE ${v}
    PROPERTY TYPE
  )
  if(_type)
    set_property(CACHE ${v} PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE ${v} PROPERTY TYPE PATH)
    endif()
  endif()
endforeach()

mark_as_advanced(ORANGEFS_ROOT ORANGEFS_LIBRARY ORANGEFS_INCLUDE_DIR)
