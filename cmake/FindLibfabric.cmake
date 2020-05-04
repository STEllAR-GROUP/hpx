# Copyright (c)      2017 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_LIBFABRIC QUIET libfabric)

find_path(
  LIBFABRIC_INCLUDE_DIR rdma/fabric.h
  HINTS ${LIBFABRIC_ROOT} ENV LIBFABRIC_ROOT ${LIBFABRIC_DIR} ENV LIBFABRIC_DIR
  PATH_SUFFIXES include
)

find_library(
  LIBFABRIC_LIBRARY
  NAMES fabric
  HINTS ${LIBFABRIC_ROOT} ENV LIBFABRIC_ROOT
  PATH_SUFFIXES lib lib64
)

# Set LIBFABRIC_ROOT in case the other hints are used
if(LIBFABRIC_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${LIBFABRIC_ROOT} LIBFABRIC_ROOT)
elseif("$ENV{LIBFABRIC_ROOT}")
  file(TO_CMAKE_PATH $ENV{LIBFABRIC_ROOT} LIBFABRIC_ROOT)
else()
  file(TO_CMAKE_PATH "${LIBFABRIC_INCLUDE_DIR}" LIBFABRIC_INCLUDE_DIR)
  string(REPLACE "/include" "" LIBFABRIC_ROOT "${LIBFABRIC_INCLUDE_DIR}")
endif()

if(NOT LIBFABRIC_INCLUDE_DIR OR NOT LIBFABRIC_LIBRARY)
  hpx_error("Could not find LIBFABRIC_INCLUDE_DIR or LIBFABRIC_LIBRARY please \
  set the LIBFABRIC_ROOT environment variable"
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Libfabric DEFAULT_MSG LIBFABRIC_LIBRARY LIBFABRIC_INCLUDE_DIR
)

mark_as_advanced(LIBFABRIC_ROOT)
