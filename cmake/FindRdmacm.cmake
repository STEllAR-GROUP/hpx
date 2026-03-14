# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_Rdmacm QUIET libibverbs)

find_path(
  Rdmacm_INCLUDE_DIR rdma/rdma_cma.h
  HINTS ${RDMACM_ROOT} ENV RDMACM_ROOT ${PC_Rdmacm_INCLUDEDIR}
        ${PC_Rdmacm_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  Rdmacm_LIBRARY
  NAMES rdmacm librdmacm
  HINTS ${RDMACM_ROOT} ENV RDMACM_ROOT ${PC_Rdmacm_LIBDIR}
        ${PC_Rdmacm_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(
  Rdmacm DEFAULT_MSG Rdmacm_LIBRARY Rdmacm_INCLUDE_DIR
)

get_property(
  _type
  CACHE Rdmacm_ROOT
  PROPERTY TYPE
)
if(_type)
  set_property(CACHE Rdmacm_ROOT PROPERTY ADVANCED 1)
  if("x${_type}" STREQUAL "xUNINITIALIZED")
    set_property(CACHE Rdmacm_ROOT PROPERTY TYPE PATH)
  endif()
endif()

mark_as_advanced(Rdmacm_ROOT Rdmacm_LIBRARY Rdmacm_INCLUDE_DIR)
