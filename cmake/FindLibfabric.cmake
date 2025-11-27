# Copyright (c)      2017 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_LIBFABRIC QUIET libfabric)

find_path(
  Libfabric_INCLUDE_DIR rdma/fabric.h
  HINTS ${LIBFABRIC_ROOT} ENV LIBFABRIC_ROOT ${Libfabric_DIR} ENV LIBFABRIC_DIR
  PATH_SUFFIXES include
)

find_library(
  Libfabric_LIBRARY
  NAMES fabric
  HINTS ${LIBFABRIC_ROOT} ENV LIBFABRIC_ROOT
  PATH_SUFFIXES lib lib64
)

get_filename_component(Libfabric_ROOT ${Libfabric_INCLUDE_DIR} DIRECTORY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Libfabric DEFAULT_MSG Libfabric_LIBRARY Libfabric_INCLUDE_DIR
)

mark_as_advanced(Libfabric_ROOT Libfabric_LIBRARY Libfabric_INCLUDE_DIR)
