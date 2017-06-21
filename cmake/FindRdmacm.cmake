# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_RDMACM QUIET libibverbs)

find_path(RDMACM_INCLUDE_DIR rdma/rdma_cma.h
  HINTS
  ${RDMACM_ROOT} ENV RDMACM_ROOT
  ${PC_RDMACM_INCLUDEDIR}
  ${PC_RDMACM_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(RDMACM_LIBRARY NAMES rdmacm librdmacm
  HINTS
    ${RDMACM_ROOT} ENV RDMACM_ROOT
    ${PC_RDMACM_LIBDIR}
    ${PC_RDMACM_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(RDMACM_LIBRARIES ${RDMACM_LIBRARY} CACHE INTERNAL "")
set(RDMACM_INCLUDE_DIRS ${RDMACM_INCLUDE_DIR} CACHE INTERNAL "")

find_package_handle_standard_args(Rdmacm DEFAULT_MSG
  RDMACM_LIBRARY RDMACM_INCLUDE_DIR)

foreach(v RDMACM_ROOT)
  get_property(_type CACHE ${v} PROPERTY TYPE)
  if(_type)
    set_property(CACHE ${v} PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE ${v} PROPERTY TYPE PATH)
    endif()
  endif()
endforeach()

mark_as_advanced(RDMACM_ROOT RDMACM_LIBRARY RDMACM_INCLUDE_DIR)
