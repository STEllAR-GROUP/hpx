# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig)
pkg_check_modules(PC_SWARM QUIET swarm)

find_path(SWARM_INCLUDE_DIR swarm/config.h
  HINTS
  ${SWARM_ROOT} ENV SWARM_ROOT
  ${PC_SWARM_INCLUDEDIR}
  ${PC_SWARM_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(SWARM_LIBRARY NAMES swarm libswarm
  HINTS
    ${SWARM_ROOT} ENV SWARM_ROOT
    ${PC_SWARM_LIBDIR}
    ${PC_SWARM_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(SWARM_LIBRARIES ${SWARM_LIBRARY})
set(SWARM_INCLUDE_DIRS ${SWARM_INCLUDE_DIR})

find_package_handle_standard_args(SWARM DEFAULT_MSG
  SWARM_LIBRARY SWARM_INCLUDE_DIR)

foreach(v SWARM_ROOT)
  get_property(_type CACHE ${v} PROPERTY TYPE)
  if(_type)
    set_property(CACHE ${v} PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE ${v} PROPERTY TYPE PATH)
    endif()
  endif()
endforeach()

mark_as_advanced(SWARM_ROOT SWARM_LIBRARY SWARM_INCLUDE_DIR)
