# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2012-2023 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET Amplifier::amplifier)
  # compatibility with older CMake versions
  if(AMPLIFIER_ROOT AND NOT Amplifier_ROOT)
    set(Amplifier_ROOT
        ${AMPLIFIER_ROOT}
        CACHE PATH "Amplifier base directory"
    )
    unset(AMPLIFIER_ROOT CACHE)
  endif()

  find_package(PkgConfig QUIET)
  pkg_check_modules(PC_AMPLIFIER QUIET amplifier)

  find_path(
    Amplifier_INCLUDE_DIR ittnotify.h
    HINTS ${Amplifier_ROOT} ENV AMPLIFIER_ROOT ${PC_Amplifier_INCLUDEDIR}
          ${PC_Amplifier_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )

  find_library(
    Amplifier_LIBRARY
    NAMES ittnotify libittnotify
    HINTS ${Amplifier_ROOT} ENV AMPLIFIER_ROOT ${PC_Amplifier_LIBDIR}
          ${PC_Amplifier_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
  )

  # Set Amplifier_ROOT in case the other hints are used
  if(Amplifier_ROOT)
    # The call to file is for compatibility for windows paths
    file(TO_CMAKE_PATH ${Amplifier_ROOT} Amplifier_ROOT)
  elseif("$ENV{AMPLIFIER_ROOT}")
    file(TO_CMAKE_PATH $ENV{AMPLIFIER_ROOT} Amplifier_ROOT)
  else()
    file(TO_CMAKE_PATH "${Amplifier_INCLUDE_DIR}" Amplifier_INCLUDE_DIR)
    string(REPLACE "/include" "" Amplifier_ROOT "${Amplifier_INCLUDE_DIR}")
  endif()

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
endif()
