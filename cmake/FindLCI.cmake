# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET LCI::LCI)
  find_package(PkgConfig QUIET)
  pkg_check_modules(PC_LCI QUIET LCI)

  find_path(
    LCI_INCLUDE_DIR lci.h
    HINTS ${LCI_ROOT} ENV LCI_ROOT ${PC_LCI_INCLUDEDIR} ${PC_LCI_INCLUDE_DIRS}
    PATH_SUFFIXES include
  )

  find_library(
    LCI_LIBRARY
    NAMES lci LCI
    HINTS ${LCI_ROOT} ENV LCI_ROOT ${PC_LCI_LIBDIR} ${PC_LCI_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
  )

  # Set LCI_ROOT in case the other hints are used
  if(NOT LCI_ROOT AND "$ENV{LCI_ROOT}")
    set(LCI_ROOT $ENV{LCI_ROOT})
  elseif(NOT LCI_ROOT)
    string(REPLACE "/include" "" LCI_ROOT "${LCI_INCLUDE_DIR}")
  endif()

  # Set LCI_ROOT in case the other hints are used
  if(LCI_ROOT)
    # The call to file is for compatibility with windows paths
    file(TO_CMAKE_PATH ${LCI_ROOT} LCI_ROOT)
  elseif("$ENV{LCI_ROOT}")
    file(TO_CMAKE_PATH $ENV{LCI_ROOT} LCI_ROOT)
  else()
    file(TO_CMAKE_PATH "${LCI_INCLUDE_DIR}" LCI_INCLUDE_DIR)
    string(REPLACE "/include" "" LCI_ROOT "${LCI_INCLUDE_DIR}")
  endif()

  set(LCI_LIBRARIES ${LCI_LIBRARY})
  set(LCI_INCLUDE_DIRS ${LCI_INCLUDE_DIR})

  find_package_handle_standard_args(LCI DEFAULT_MSG LCI_LIBRARY LCI_INCLUDE_DIR)

  get_property(
    _type
    CACHE LCI_ROOT
    PROPERTY TYPE
  )
  if(_type)
    set_property(CACHE LCI_ROOT PROPERTY ADVANCED 1)
    if("x${_type}" STREQUAL "xUNINITIALIZED")
      set_property(CACHE LCI_ROOT PROPERTY TYPE PATH)
    endif()
  endif()

  mark_as_advanced(LCI_ROOT LCI_LIBRARY LCI_INCLUDE_DIR)

  add_library(LCI::LCI INTERFACE IMPORTED)
  target_include_directories(LCI::LCI SYSTEM INTERFACE ${LCI_INCLUDE_DIR})
  target_link_libraries(LCI::LCI INTERFACE ${LCI_LIBRARY})
endif()
