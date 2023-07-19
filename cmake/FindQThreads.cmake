# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# compatibility with older CMake versions
if(QTHREADS_ROOT AND NOT Qthreads_ROOT)
  set(Qthreads_ROOT
      ${QTHREADS_ROOT}
      CACHE PATH "QThreads base directory"
  )
  unset(QTHREADS_ROOT CACHE)
endif()

find_package(PkgConfig QUIET)
pkg_check_modules(PC_Qthreads QUIET swarm)

find_path(
  Qthreads_INCLUDE_DIR qthread/qthread.h
  HINTS ${Qthreads_ROOT} ENV QTHREADS_ROOT ${PC_Qthreads_INCLUDEDIR}
        ${PC_Qthreads_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  Qthreads_LIBRARY
  NAMES qthread libqthread
  HINTS ${Qthreads_ROOT} ENV QTHREADS_ROOT ${PC_Qthreads_LIBDIR}
        ${PC_Qthreads_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set Qthreads_ROOT in case the other hints are used
if(Qthreads_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${Qthreads_ROOT} Qthreads_ROOT)
elseif("$ENV{QTHREADS_ROOT}")
  file(TO_CMAKE_PATH $ENV{QTHREADS_ROOT} Qthreads_ROOT)
else()
  file(TO_CMAKE_PATH "${Qthreads_INCLUDE_DIR}" Qthreads_INCLUDE_DIR)
  string(REPLACE "/include" "" Qthreads_ROOT "${Qthreads_INCLUDE_DIR}")
endif()

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
