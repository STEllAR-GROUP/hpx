# Copyright (c)      2017 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig)
pkg_check_modules(PC_LIBFABRIC QUIET libfabric)

find_path(LIBFABRIC_INCLUDE_DIR rdma/fabric.h
  HINTS
    ${LIBFABRIC_ROOT} ENV LIBFABRIC_ROOT
    ${LIBFABRIC_DIR} ENV LIBFABRIC_DIR
  PATH_SUFFIXES include)

find_library(LIBFABRIC_LIBRARY NAMES fabric
  HINTS
    ${LIBFABRIC_ROOT} ENV LIBFABRIC_ROOT
  PATH_SUFFIXES lib lib64)

set(LIBFABRIC_LIBRARIES ${LIBFABRIC_LIBRARY} CACHE INTERNAL "")
set(LIBFABRIC_INCLUDE_DIRS ${LIBFABRIC_INCLUDE_DIR} CACHE INTERNAL "")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Libfabric DEFAULT_MSG
  LIBFABRIC_LIBRARY LIBFABRIC_INCLUDE_DIR)

#foreach(v LIBFABRIC_ROOT)
#  get_property(_type CACHE ${v} PROPERTY TYPE)
#  if(_type)
#    set_property(CACHE ${v} PROPERTY ADVANCED 1)
#    if("x${_type}" STREQUAL "xUNINITIALIZED")
#      set_property(CACHE ${v} PROPERTY TYPE PATH)
#    endif()
#  endif()
#endforeach()

mark_as_advanced(LIBFABRIC_ROOT LIBFABRIC_LIBRARY LIBFABRIC_INCLUDE_DIR)
