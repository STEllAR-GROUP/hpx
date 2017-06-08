# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_TBBMALLOC QUIET libtbbmalloc)

find_path(TBBMALLOC_INCLUDE_DIR tbb/scalable_allocator.h
  HINTS
    ${TBBMALLOC_ROOT} ENV TBBMALLOC_ROOT
    ${PC_TBBMALLOC_INCLUDEDIR}
    ${PC_TBBMALLOC_INCLUDE_DIRS}
  PATH_SUFFIXES include)

set(TBBMALLOC_PATH_SUFFIX "lib/intel64" "lib/intel64/gcc4.4")
if(TBBMALLOC_PLATFORM STREQUAL "mic")
  set(TBBMALLOC_PATH_SUFFIX "lib/mic")
endif()
if(TBBMALLOC_PLATFORM STREQUAL "mic-knl")
  set(TBBMALLOC_PATH_SUFFIX "lib/intel64_lin_mic")
endif()

message("${TBBMALLOC_ROOT} ${TBBMALLOC_PATH_SUFFIX} ${TBBMALLOC_PLATFORM}")

find_library(TBBMALLOC_LIBRARY NAMES tbbmalloc libtbbmalloc
  HINTS
    ${TBBMALLOC_ROOT} ENV TBBMALLOC_ROOT
    ${PC_TBBMALLOC_LIBDIR}
    ${PC_TBBMALLOC_LIBRARY_DIRS}
  PATH_SUFFIXES ${TBBMALLOC_PATH_SUFFIX} lib lib64)

find_library(TBBMALLOC_PROXY_LIBRARY NAMES tbbmalloc_proxy libtbbmalloc_proxy
  HINTS
    ${TBBMALLOC_ROOT} ENV TBBMALLOC_ROOT
    ${PC_TBBMALLOC_LIBDIR}
    ${PC_TBBMALLOC_LIBRARY_DIRS}
  PATH_SUFFIXES ${TBBMALLOC_PATH_SUFFIX} lib lib64)

set(TBBMALLOC_LIBRARIES ${TBBMALLOC_LIBRARY} ${TBBMALLOC_PROXY_LIBRARY})
set(TBBMALLOC_INCLUDE_DIRS ${TBBMALLOC_INCLUDE_DIR})

find_package_handle_standard_args(TBBmalloc DEFAULT_MSG
  TBBMALLOC_LIBRARY TBBMALLOC_PROXY_LIBRARY TBBMALLOC_INCLUDE_DIR)

foreach(v TBBMALLOC_ROOT TBBMALLOC_PLATFORM)
  get_property(_type CACHE ${v} PROPERTY TYPE)
  if(_type)
    set_property(CACHE ${v} PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE ${v} PROPERTY TYPE PATH)
    endif()
  endif()
endforeach()

mark_as_advanced(TBBMALLOC_ROOT TBBMALLOC_LIBRARY TBBMALLOC_PROXY_LIBRARY TBBMALLOC_INCLUDE_DIR)
