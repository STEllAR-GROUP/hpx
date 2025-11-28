# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_Qthreads QUIET swarm)

find_path(
  Qthreads_INCLUDE_DIR qthread/qthread.h
  HINTS ${QTHREADS_ROOT} ENV QTHREADS_ROOT ${PC_Qthreads_INCLUDEDIR}
        ${PC_Qthreads_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  Qthreads_LIBRARY
  NAMES qthread libqthread
  HINTS ${QTHREADS_ROOT} ENV QTHREADS_ROOT ${PC_Qthreads_LIBDIR}
        ${PC_Qthreads_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

get_filename_component(Qthreads_ROOT ${Qthreads_INCLUDE_DIR} DIRECTORY)
set(Qthreads_LIBRARIES ${Qthreads_LIBRARY})
set(Qthreads_INCLUDE_DIRS ${Qthreads_INCLUDE_DIR})

find_package_handle_standard_args(
  QThreads DEFAULT_MSG Qthreads_LIBRARY Qthreads_INCLUDE_DIR
)

foreach(v Qthreads_ROOT)
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

mark_as_advanced(Qthreads_ROOT Qthreads_LIBRARY Qthreads_INCLUDE_DIR)
