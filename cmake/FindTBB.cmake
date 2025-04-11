# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2011-2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# compatibility with older CMake versions
if(TBB_ROOT AND NOT Tbb_ROOT)
  set(Tbb_ROOT
      ${TBB_ROOT}
      CACHE PATH "TBB base directory"
  )
  unset(TBB_ROOT CACHE)
endif()

find_package(PkgConfig QUIET)
pkg_check_modules(PC_Tbb QUIET libtbb)

find_path(
  Tbb_INCLUDE_DIR tbb/tbb.h
  HINTS ${Tbb_ROOT} ENV TBB_ROOT ${PC_Tbb_INCLUDEDIR} ${PC_Tbb_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

set(Tbb_PATH_SUFFIX "lib/intel64" "lib/intel64/gcc4.4")
if(Tbb_PLATFORM STREQUAL "mic")
  set(Tbb_PATH_SUFFIX "lib/mic")
endif()
if(Tbb_PLATFORM STREQUAL "mic-knl")
  set(Tbb_PATH_SUFFIX "lib/intel64_lin_mic")
endif()

find_library(
  Tbb_PROXY_LIBRARY
  NAMES tbb libtbb
  HINTS ${Tbb_ROOT} ENV TBB_ROOT ${PC_Tbb_LIBDIR} ${PC_Tbb_LIBRARY_DIRS}
  PATH_SUFFIXES ${Tbb_PATH_SUFFIX} lib lib64
)

set(Tbb_LIBRARIES ${Tbb_LIBRARY} ${Tbb_PROXY_LIBRARY})
set(Tbb_INCLUDE_DIRS ${Tbb_INCLUDE_DIR})

find_package_handle_standard_args(
  TBB DEFAULT_MSG TBB_LIBRARY TBB_INCLUDE_DIR
)

foreach(v Tbb_ROOT Tbb_PLATFORM)
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

mark_as_advanced(Tbb_ROOT Tbb_LIBRARY Tbb_PROXY_LIBRARY Tbb_INCLUDE_DIR)


find_path(Tbb_INCLUDE_DIR tbb/tbb.h PATHS /usr/include)
find_library(TBB_LIBRARY NAMES tbb PATHS /usr/lib /usr/lib/x86_64-linux-gnu)
