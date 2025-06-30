# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2012-2023 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET Amplifier::amplifier)

  find_package(PkgConfig QUIET)
  pkg_check_modules(PC_AMPLIFIER QUIET amplifier)

  find_path(
    Amplifier_INCLUDE_DIR ittnotify.h
    HINTS ${AMPLIFIER_ROOT} ENV AMPLIFIER_ROOT ${PC_Amplifier_INCLUDEDIR}
          ${PC_Amplifier_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )

  find_library(
    Amplifier_LIBRARY
    NAMES ittnotify libittnotify
    HINTS ${AMPLIFIER_ROOT} ENV AMPLIFIER_ROOT ${PC_Amplifier_LIBDIR}
          ${PC_Amplifier_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
  )

  set(Amplifier_LIBRARIES ${Amplifier_LIBRARY})
  set(Amplifier_INCLUDE_DIRS ${Amplifier_INCLUDE_DIR})

  find_package_handle_standard_args(
    Amplifier DEFAULT_MSG Amplifier_LIBRARY Amplifier_INCLUDE_DIR
  )

  get_property(
    _type
    CACHE Amplifier_ROOT
    PROPERTY TYPE
  )
  if(_type)
    set_property(CACHE Amplifier_ROOT PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE Amplifier_ROOT PROPERTY TYPE PATH)
    endif()
  endif()

  mark_as_advanced(Amplifier_ROOT Amplifier_LIBRARY Amplifier_INCLUDE_DIR)

  add_library(Amplifier::amplifier INTERFACE IMPORTED)
  target_include_directories(
    Amplifier::amplifier SYSTEM INTERFACE ${Amplifier_INCLUDE_DIR}
  )
  target_link_libraries(Amplifier::amplifier INTERFACE ${Amplifier_LIBRARIES})
  mark_as_advanced(Amplifier_LIBRARIES, Amplifier_INCLUDE_DIRS)

  # Set Amplifier_ROOT
  get_filename_component(Amplifier_ROOT ${Amplifier_INCLUDE_DIR} DIRECTORY)
endif()
