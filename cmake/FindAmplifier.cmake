# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_AMPLIFIER QUIET amplifier)

find_path(AMPLIFIER_INCLUDE_DIR ittnotify.h
  HINTS
    ${AMPLIFIER_ROOT} ENV AMPLIFIER_ROOT
    ${PC_AMPLIFIER_INCLUDEDIR}
    ${PC_AMPLIFIER_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(AMPLIFIER_LIBRARY NAMES ittnotify libittnotify
  HINTS
    ${AMPLIFIER_ROOT} ENV AMPLIFIER_ROOT
    ${PC_AMPLIFIER_LIBDIR}
    ${PC_AMPLIFIER_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(AMPLIFIER_LIBRARIES ${AMPLIFIER_LIBRARY})
set(AMPLIFIER_INCLUDE_DIRS ${AMPLIFIER_INCLUDE_DIR})

find_package_handle_standard_args(Amplifier DEFAULT_MSG
  AMPLIFIER_LIBRARY AMPLIFIER_INCLUDE_DIR)

get_property(_type CACHE AMPLIFIER_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE AMPLIFIER_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE AMPLIFIER_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(AMPLIFIER_ROOT AMPLIFIER_LIBRARY AMPLIFIER_INCLUDE_DIR)
