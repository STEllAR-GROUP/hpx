# Copyright (c)      2017 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# compatibility with older CMake versions
if(LIBFABRIC_ROOT AND NOT Libfabric_ROOT)
  set(Libfabric_ROOT
      ${LIBFABRIC_ROOT}
      CACHE PATH "Libfabric base directory"
  )
  unset(LIBFABRIC_ROOT CACHE)
endif()

find_package(PkgConfig QUIET)
pkg_check_modules(PC_LIBFABRIC QUIET libfabric)

find_path(
  Libfabric_INCLUDE_DIR rdma/fabric.h
  HINTS ${Libfabric_ROOT} ENV LIBFABRIC_ROOT ${Libfabric_DIR} ENV LIBFABRIC_DIR
  PATH_SUFFIXES include
)

find_library(
  Libfabric_LIBRARY
  NAMES fabric
  HINTS ${Libfabric_ROOT} ENV LIBFABRIC_ROOT
  PATH_SUFFIXES lib lib64
)

# Set Libfabric_ROOT in case the other hints are used
if(Libfabric_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${Libfabric_ROOT} Libfabric_ROOT)
elseif("$ENV{LIBFABRIC_ROOT}")
  file(TO_CMAKE_PATH $ENV{LIBFABRIC_ROOT} Libfabric_ROOT)
else()
  file(TO_CMAKE_PATH "${Libfabric_INCLUDE_DIR}" Libfabric_INCLUDE_DIR)
  string(REPLACE "/include" "" Libfabric_ROOT "${Libfabric_INCLUDE_DIR}")
endif()

if(NOT Libfabric_INCLUDE_DIR OR NOT Libfabric_LIBRARY)
  hpx_error("Could not find Libfabric_INCLUDE_DIR or Libfabric_LIBRARY please \
  set the Libfabric_ROOT environment variable"
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Libfabric DEFAULT_MSG Libfabric_LIBRARY Libfabric_INCLUDE_DIR
)

mark_as_advanced(Libfabric_ROOT)
