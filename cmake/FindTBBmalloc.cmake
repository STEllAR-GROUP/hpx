# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_Tbbmalloc QUIET libtbbmalloc)

find_path(
  Tbbmalloc_INCLUDE_DIR tbb/scalable_allocator.h
  HINTS ${TBBMALLOC_ROOT} ENV TBBMALLOC_ROOT ${HPX_TBBMALLOC_ROOT}
        ${PC_Tbbmalloc_INCLUDEDIR} ${PC_Tbbmalloc_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

set(Tbbmalloc_PATH_SUFFIX "lib/intel64" "lib/intel64/gcc4.4")
if(Tbbmalloc_PLATFORM STREQUAL "mic")
  set(Tbbmalloc_PATH_SUFFIX "lib/mic")
endif()
if(Tbbmalloc_PLATFORM STREQUAL "mic-knl")
  set(Tbbmalloc_PATH_SUFFIX "lib/intel64_lin_mic")
endif()

find_library(
  Tbbmalloc_LIBRARY
  NAMES tbbmalloc libtbbmalloc
  HINTS ${TBBMALLOC_ROOT} ENV TBBMALLOC_ROOT ${HPX_TBBMALLOC_ROOT}
        ${PC_Tbbmalloc_LIBDIR} ${PC_Tbbmalloc_LIBRARY_DIRS}
)

find_library(
  Tbbmalloc_PROXY_LIBRARY
  NAMES tbbmalloc_proxy libtbbmalloc_proxy
  HINTS ${TBBMALLOC_ROOT} ENV TBBMALLOC_ROOT ${HPX_TBBMALLOC_ROOT}
        ${PC_Tbbmalloc_LIBDIR} ${PC_Tbbmalloc_LIBRARY_DIRS}
)

# Set Tbbmalloc_ROOT
get_filename_component(Tbbmalloc_ROOT ${Tbbmalloc_INCLUDE_DIR} DIRECTORY)

set(Tbbmalloc_LIBRARIES ${Tbbmalloc_LIBRARY} ${Tbbmalloc_PROXY_LIBRARY})
set(Tbbmalloc_INCLUDE_DIRS ${Tbbmalloc_INCLUDE_DIR})

find_package_handle_standard_args(
  TBBmalloc DEFAULT_MSG Tbbmalloc_LIBRARY Tbbmalloc_PROXY_LIBRARY
  Tbbmalloc_INCLUDE_DIR
)

foreach(v Tbbmalloc_ROOT Tbbmalloc_PLATFORM)
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

mark_as_advanced(
  Tbbmalloc_ROOT Tbbmalloc_LIBRARY Tbbmalloc_PROXY_LIBRARY
  Tbbmalloc_INCLUDE_DIR
)
