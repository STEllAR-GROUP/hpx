# Copyright (c) 2020      ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_MIMALLOC QUIET libmimalloc)

find_library(MIMALLOC_LIBRARY NAMES mimalloc libmimalloc
  HINTS
    $ENV{MIMALLOC_ROOT}
    ${MIMALLOC_ROOT}
    ${HPX_MIMALLOC_ROOT}
    ${PC_MIMALLOC_LIBDIR}
    ${PC_MIMALLOC_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

if (MIMALLOC_LIBRARY)
  set(Mimalloc_FOUND "Found")
endif()

# Set MIMALLOC_ROOT in case the other hints are used
if (NOT MIMALLOC_ROOT AND "$ENV{MIMALLOC_ROOT}")
  set(MIMALLOC_ROOT $ENV{MIMALLOC_ROOT})
elseif(NOT MIMALLOC_ROOT)
  string(REGEX REPLACE "/lib/*" "" MIMALLOC_ROOT "${MIMALLOC_LIBRARY}")
endif()

set(MIMALLOC_LIBRARIES ${MIMALLOC_LIBRARY})

get_property(_type CACHE MIMALLOC_ROOT PROPERTY TYPE)
if(_type)
  set_property(CACHE MIMALLOC_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE MIMALLOC_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(MIMALLOC_ROOT MIMALLOC_LIBRARY)
