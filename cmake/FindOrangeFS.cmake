# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2013      Shuangyang Yang
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# compatibility with older CMake versions
if(ORANGEFS_ROOT AND NOT Orangefs_ROOT)
  set(Orangefs_ROOT
      ${ORANGEFS_ROOT}
      CACHE PATH "OrangeFS base directory"
  )
  unset(ORANGEFS_ROOT CACHE)
endif()

find_package(PkgConfig QUIET)
pkg_check_modules(PC_Orangefs QUIET orangefs)

find_path(
  Orangefs_INCLUDE_DIR
  NAMES pxfs.h orange.h
  HINTS ${Orangefs_ROOT} ENV Orangefs_ROOT ${PC_Orangefs_INCLUDEDIR}
        ${PC_Orangefs_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  Orangefs_LIBRARY
  NAMES pvfs2 # orangefs pvfs2 orangefsposix
  HINTS ${Orangefs_ROOT} ENV ORANGEFS_ROOT ${PC_Orangefs_LIBDIR}
        ${PC_Orangefs_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set Orangefs_ROOT in case the other hints are used
if(Orangefs_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${Orangefs_ROOT} Orangefs_ROOT)
elseif("$ENV{ORANGEFS_ROOT}")
  file(TO_CMAKE_PATH $ENV{ORANGEFS_ROOT} Orangefs_ROOT)
else()
  file(TO_CMAKE_PATH "${Orangefs_INCLUDE_DIR}" Orangefs_INCLUDE_DIR)
  string(REPLACE "/include" "" Orangefs_ROOT "${Orangefs_INCLUDE_DIR}")
endif()

set(Orangefs_LIBRARIES ${Orangefs_LIBRARY})
set(Orangefs_INCLUDE_DIRS ${Orangefs_INCLUDE_DIR})

find_package_handle_standard_args(
  OrangeFS DEFAULT_MSG Orangefs_LIBRARY Orangefs_INCLUDE_DIR
)

foreach(v Orangefs_ROOT)
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

mark_as_advanced(Orangefs_ROOT Orangefs_LIBRARY Orangefs_INCLUDE_DIR)
