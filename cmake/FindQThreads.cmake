# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_QTHREADS QUIET swarm)

find_path(
  QTHREADS_INCLUDE_DIR qthread/qthread.h
  HINTS ${QTHREADS_ROOT} ENV QTHREADS_ROOT ${PC_QTHREADS_INCLUDEDIR}
        ${PC_QTHREADS_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  QTHREADS_LIBRARY
  NAMES qthread libqthread
  HINTS ${QTHREADS_ROOT} ENV QTHREADS_ROOT ${PC_QTHREADS_LIBDIR}
        ${PC_QTHREADS_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set QTHREADS_ROOT in case the other hints are used
if(QTHREADS_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${QTHREADS_ROOT} QTHREADS_ROOT)
elseif("$ENV{QTHREADS_ROOT}")
  file(TO_CMAKE_PATH $ENV{QTHREADS_ROOT} QTHREADS_ROOT)
else()
  file(TO_CMAKE_PATH "${QTHREADS_INCLUDE_DIR}" QTHREADS_INCLUDE_DIR)
  string(REPLACE "/include" "" QTHREADS_ROOT "${QTHREADS_INCLUDE_DIR}")
endif()

set(QTHREADS_LIBRARIES ${QTHREADS_LIBRARY})
set(QTHREADS_INCLUDE_DIRS ${QTHREADS_INCLUDE_DIR})

find_package_handle_standard_args(
  QThreads DEFAULT_MSG QTHREADS_LIBRARY QTHREADS_INCLUDE_DIR
)

foreach(v QTHREADS_ROOT)
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

mark_as_advanced(QTHREADS_ROOT QTHREADS_LIBRARY QTHREADS_INCLUDE_DIR)
