# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2012 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET Amplifier::amplifier)
  find_package(PkgConfig QUIET)
  pkg_check_modules(PC_AMPLIFIER QUIET amplifier)

  find_path(
    AMPLIFIER_INCLUDE_DIR ittnotify.h
    HINTS ${AMPLIFIER_ROOT} ENV AMPLIFIER_ROOT ${PC_AMPLIFIER_INCLUDEDIR}
          ${PC_AMPLIFIER_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )

  find_library(
    AMPLIFIER_LIBRARY
    NAMES ittnotify libittnotify
    HINTS ${AMPLIFIER_ROOT} ENV AMPLIFIER_ROOT ${PC_AMPLIFIER_LIBDIR}
          ${PC_AMPLIFIER_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
  )

  # Set AMPLIFIER_ROOT in case the other hints are used
  if(AMPLIFIER_ROOT)
    # The call to file is for compatibility for windows paths
    file(TO_CMAKE_PATH ${AMPLIFIER_ROOT} AMPLIFIER_ROOT)
  elseif("$ENV{AMPLIFIER_ROOT}")
    file(TO_CMAKE_PATH $ENV{AMPLIFIER_ROOT} AMPLIFIER_ROOT)
  else()
    file(TO_CMAKE_PATH "${AMPLIFIER_INCLUDE_DIR}" AMPLIFIER_INCLUDE_DIR)
    string(REPLACE "/include" "" AMPLIFIER_ROOT "${AMPLIFIER_INCLUDE_DIR}")
  endif()

  set(AMPLIFIER_LIBRARIES ${AMPLIFIER_LIBRARY})
  set(AMPLIFIER_INCLUDE_DIRS ${AMPLIFIER_INCLUDE_DIR})

  find_package_handle_standard_args(
    Amplifier DEFAULT_MSG AMPLIFIER_LIBRARY AMPLIFIER_INCLUDE_DIR
  )

  get_property(
    _type
    CACHE AMPLIFIER_ROOT
    PROPERTY TYPE
  )
  if(_type)
    set_property(CACHE AMPLIFIER_ROOT PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE AMPLIFIER_ROOT PROPERTY TYPE PATH)
    endif()
  endif()

  mark_as_advanced(AMPLIFIER_ROOT AMPLIFIER_LIBRARY AMPLIFIER_INCLUDE_DIR)

  add_library(Amplifier::amplifier INTERFACE IMPORTED)
  target_include_directories(
    Amplifier::amplifier SYSTEM INTERFACE ${AMPLIFIER_INCLUDE_DIR}
  )
  target_link_libraries(Amplifier::amplifier INTERFACE ${AMPLIFIER_LIBRARIES})
endif()
