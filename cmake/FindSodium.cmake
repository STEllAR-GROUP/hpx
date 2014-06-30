# Copyright (c) 2014 Thomas Heller
# Copyright (c) 2013 Jeroen Habraken
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig)
pkg_check_modules(PC_SODIUM QUIET sodium)

find_path(SODIUM_INCLUDE_DIR sodium.h
  HINTS
    ${SODIUM_ROOT} ENV SODIUM_ROOT
    ${PC_SODIUM_INCLUDEDIR}
    ${PC_SODIUM_INCLUDE_DIRS}
  PATH_SUFFIXES include src/libsodium/include)

find_library(SODIUM_LIBRARY NAMES sodium libsodium
  HINTS
    ${SODIUM_ROOT} ENV SODIUM_ROOT
    ${PC_SODIUM_LIBDIR}
    ${PC_SODIUM_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64 src/libsodium/.libs)

set(SODIUM_LIBRARIES ${SODIUM_LIBRARY})
set(SODIUM_INCLUDE_DIRS ${SODIUM_INCLUDE_DIR})

find_package_handle_standard_args(Sodium DEFAULT_MSG
  SODIUM_LIBRARY SODIUM_INCLUDE_DIR)

get_property(_type CACHE SODIUM_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE SODIUM_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE SODIUM_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(SODIUM_ROOT SODIUM_LIBRARY SODIUM_INCLUDE_DIR)
