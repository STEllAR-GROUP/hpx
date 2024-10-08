# Copyright (c)      2017 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# compatibility with older CMake versions
if(PMI_ROOT AND NOT Pmi_ROOT)
  set(Pmi_ROOT
      ${PMI_ROOT}
      CACHE PATH "PMI base directory"
  )
  unset(PMI_ROOT CACHE)
endif()

find_package(PkgConfig QUIET)

# look for cray pmi...
pkg_check_modules(PC_Pmi_CRAY QUIET cray-pmi)

# look for the rest if we couldn't find the cray package
if(NOT PC_Pmi_CRAY_FOUND)
  pkg_check_modules(PC_Pmi QUIET pmi)
endif()

find_path(
  Pmi_INCLUDE_DIR pmi2.h
  HINTS ${Pmi_ROOT}
        ENV
        PMI_ROOT
        ${Pmi_DIR}
        ENV
        PMI_DIR
        ${PC_Pmi_CRAY_INCLUDEDIR}
        ${PC_Pmi_CRAY_INCLUDE_DIRS}
        ${PC_Pmi_INCLUDEDIR}
        ${PC_Pmi_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

find_library(
  Pmi_LIBRARY
  NAMES pmi
  HINTS ${Pmi_ROOT}
        ENV
        PMI_ROOT
        ${PC_Pmi_CRAY_LIBDIR}
        ${PC_Pmi_CRAY_LIBRARY_DIRS}
        ${PC_Pmi_LIBDIR}
        ${PC_Pmi_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64
)

# Set Pmi_ROOT in case the other hints are used
if(Pmi_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${Pmi_ROOT} Pmi_ROOT)
elseif(DEFINED ENV{PMI_ROOT})
  file(TO_CMAKE_PATH $ENV{PMI_ROOT} Pmi_ROOT)
else()
  file(TO_CMAKE_PATH "${Pmi_INCLUDE_DIR}" Pmi_INCLUDE_DIR)
  string(REPLACE "/include" "" Pmi_ROOT "${Pmi_INCLUDE_DIR}")
endif()

if(NOT Pmi_LIBRARY OR NOT Pmi_INCLUDE_DIR)
  set(Pmi_FOUND=OFF)
  return()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PMI DEFAULT_MSG Pmi_LIBRARY Pmi_INCLUDE_DIR)

mark_as_advanced(Pmi_ROOT Pmi_LIBRARY Pmi_INCLUDE_DIR)
