# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig)
pkg_check_modules(PC_TAU QUIET TAU)

# This if statement is specific to TAU, and should not be copied into other
# Find cmake scripts.
if(NOT TAU_ROOT AND NOT $ENV{TAU_ROOT} STREQUAL "")
  set(TAU_ROOT "$ENV{TAU_ROOT}")
endif()

if(NOT TAU_ARCH AND NOT $ENV{TAU_ARCH} STREQUAL "")
    set(TAU_ARCH $ENV{TAU_ARCH})
endif()

if(NOT TAU_ARCH AND NOT $ENV{TAU_OPTIONS} STREQUAL "")
    set(TAU_ARCH $ENV{TAU_OPTIONS})
endif()

find_path(TAU_INCLUDE_DIR TAU.h
  HINTS
    ${TAU_ROOT}/include ENV TAU_ROOT
    ${PC_TAU_INCLUDEDIR}
    ${PC_TAU_INCLUDE_DIRS}
  PATH_SUFFIXES include)

if (${APPLE})
  find_library(TAU_LIBRARY NAMES TAUsh${TAU_OPTIONS} tau${TAU_OPTIONS} TAU
             HINTS ${TAU_ROOT}/apple/lib)
  find_path(TAU_LIBRARY_DIR NAMES TAUsh${TAU_OPTIONS}.so tau${TAU_OPTIONS}.a libTAU.dylib
             HINTS ${TAU_ROOT}/apple/lib)
else()
    find_library(TAU_LIBRARY NAMES TAUsh${TAU_OPTIONS} tau${TAU_OPTIONS} TAU
      HINTS ${TAU_ROOT}/${TAU_ARCH}/lib ${TAU_ROOT}/${CMAKE_SYSTEM_PROCESSOR}/lib  ${TAU_ROOT}/*/lib )
  find_path(TAU_LIBRARY_DIR NAMES tau${TAU_OPTIONS}.a TAUsh${TAU_OPTIONS}.so libTAU.so libTAU.a libTAU.dylib
      HINTS ${TAU_ROOT}/${TAU_ARCH}/lib ${TAU_ROOT}/${CMAKE_SYSTEM_PROCESSOR}/lib  ${TAU_ROOT}/*/lib )
endif()

set(TAU_LIBRARIES ${TAU_LIBRARY} m)
set(TAU_INCLUDE_DIRS ${TAU_INCLUDE_DIR})

find_package_handle_standard_args(TAU DEFAULT_MSG
  TAU_LIBRARY TAU_INCLUDE_DIR)

get_property(_type CACHE TAU_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE TAU_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE TAU_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(TAU_ROOT TAU_LIBRARY TAU_INCLUDE_DIR)
