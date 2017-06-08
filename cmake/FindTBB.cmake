# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2011-2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_TBB QUIET libtbb)

find_path(TBB_INCLUDE_DIR tbb/tbb.h
  HINTS
    ${TBB_ROOT} ENV TBB_ROOT
    ${PC_TBB_INCLUDEDIR}
    ${PC_TBB_INCLUDE_DIRS}
  PATH_SUFFIXES include)

set(TBB_PATH_SUFFIX "lib/intel64" "lib/intel64/gcc4.4")
if(TBB_PLATFORM STREQUAL "mic")
  set(TBB_PATH_SUFFIX "lib/mic")
endif()
if(TBB_PLATFORM STREQUAL "mic-knl")
  set(TBB_PATH_SUFFIX "lib/intel64_lin_mic")
endif()

find_library(TBB_PROXY_LIBRARY NAMES tbb libtbb
  HINTS
    ${TBB_ROOT} ENV TBB_ROOT
    ${PC_TBB_LIBDIR}
    ${PC_TBB_LIBRARY_DIRS}
  PATH_SUFFIXES ${TBB_PATH_SUFFIX} lib lib64)

set(TBB_LIBRARIES ${TBB_LIBRARY} ${TBB_PROXY_LIBRARY})
set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})

find_package_handle_standard_args(TBBmalloc DEFAULT_MSG
  TBB_LIBRARY TBB_PROXY_LIBRARY TBB_INCLUDE_DIR)

foreach(v TBB_ROOT TBB_PLATFORM)
  get_property(_type CACHE ${v} PROPERTY TYPE)
  if(_type)
    set_property(CACHE ${v} PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE ${v} PROPERTY TYPE PATH)
    endif()
  endif()
endforeach()

mark_as_advanced(TBB_ROOT TBB_LIBRARY TBB_PROXY_LIBRARY TBB_INCLUDE_DIR)
