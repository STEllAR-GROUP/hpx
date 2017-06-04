# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig QUIET)
pkg_check_modules(PC_IBVERBS QUIET libibverbs)

find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h
  HINTS
  ${IBVERBS_ROOT} ENV IBVERBS_ROOT
  ${PC_IBVERBS_INCLUDEDIR}
  ${PC_IBVERBS_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(IBVERBS_LIBRARY NAMES ibverbs libibverbs
  HINTS
    ${IBVERBS_ROOT} ENV IBVERBS_ROOT
    ${PC_IBVERBS_LIBDIR}
    ${PC_IBVERBS_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(IBVERBS_LIBRARIES ${IBVERBS_LIBRARY} CACHE INTERNAL "")
set(IBVERBS_INCLUDE_DIRS ${IBVERBS_INCLUDE_DIR} CACHE INTERNAL "")

find_package_handle_standard_args(Ibverbs DEFAULT_MSG
  IBVERBS_LIBRARY IBVERBS_INCLUDE_DIR)

foreach(v IBVERBS_ROOT)
  get_property(_type CACHE ${v} PROPERTY TYPE)
  if(_type)
    set_property(CACHE ${v} PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE ${v} PROPERTY TYPE PATH)
    endif()
  endif()
endforeach()

mark_as_advanced(IBVERBS_ROOT IBVERBS_LIBRARY IBVERBS_INCLUDE_DIR)
