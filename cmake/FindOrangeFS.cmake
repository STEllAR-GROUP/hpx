# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2013      Shuangyang Yang
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(PkgConfig)
pkg_check_modules(PC_ORANGESF QUIET orangefs)

find_path(ORANGESF_INCLUDE_DIR NAMES pxfs.h orange.h
  HINTS
  ${ORANGESF_ROOT} ENV ORANGESF_ROOT
  ${PC_ORANGESF_INCLUDEDIR}
  ${PC_ORANGESF_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(ORANGESF_LIBRARY NAMES pvfs2 # orangefs pvfs2 orangefsposix
  HINTS
    ${ORANGESF_ROOT} ENV ORANGESF_ROOT
    ${PC_ORANGESF_LIBDIR}
    ${PC_ORANGESF_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(ORANGESF_LIBRARIES ${ORANGESF_LIBRARY})
set(ORANGESF_INCLUDE_DIRS ${ORANGESF_INCLUDE_DIR})

find_package_handle_standard_args(OrangeFS DEFAULT_MSG
  ORANGESF_LIBRARY ORANGESF_INCLUDE_DIR)

foreach(v ORANGESF_ROOT)
  get_property(_type CACHE ${v} PROPERTY TYPE)
  if(_type)
    set_property(CACHE ${v} PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE ${v} PROPERTY TYPE PATH)
    endif()
  endif()
endforeach()

mark_as_advanced(ORANGESF_ROOT ORANGESF_LIBRARY ORANGESF_INCLUDE_DIR)
