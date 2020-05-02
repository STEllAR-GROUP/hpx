# - Try to find BZip2
# Once done this will define
#
#  BZIP2_FOUND - system has BZip2
#  BZIP2_INCLUDE_DIR - the BZip2 include directory
#  BZIP2_LIBRARIES - Link these to use BZip2
#  BZIP2_NEED_PREFIX - this is set if the functions are prefixed with BZ2_
#  BZIP2_VERSION_STRING - the version of BZip2 found (since CMake 2.8.8)

# =============================================================================
# Copyright 2006-2012 Kitware, Inc. Copyright 2006 Alexander Neundorf
# <neundorf@kde.org> Copyright 2012 Rolf Eike Beer <eike@sf-mail.de>
#
# SPDX-License-Identifier: BSL-1.0 Distributed under the OSI-approved BSD
# License (the "License"); see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# License for more information.
#
# Make the HPX inspect tool happy: hpxinspect:nolicense
#
# =============================================================================
# (To distribute this file outside of CMake, substitute the full License text
# for the above reference.)

set(_BZIP2_PATHS PATHS
                 "[HKEY_LOCAL_MACHINE\\SOFTWARE\\GnuWin32\\Bzip2;InstallPath]"
)

find_path(BZIP2_INCLUDE_DIR bzlib.h ${_BZIP2_PATHS} PATH_SUFFIXES include)

if(NOT BZIP2_LIBRARIES)
  find_library(
    BZIP2_LIBRARY_RELEASE
    NAMES bz2 bzip2 ${_BZIP2_PATHS}
    PATH_SUFFIXES lib
  )
  find_library(
    BZIP2_LIBRARY_DEBUG
    NAMES bzip2d ${_BZIP2_PATHS}
    PATH_SUFFIXES lib
  )

  include(SelectLibraryConfigurations)
  select_library_configurations(BZIP2)
endif()

if(BZIP2_INCLUDE_DIR AND EXISTS "${BZIP2_INCLUDE_DIR}/bzlib.h")
  file(STRINGS "${BZIP2_INCLUDE_DIR}/bzlib.h" BZLIB_H
       REGEX "bzip2/libbzip2 version [0-9]+\\.[^ ]+ of [0-9]+ "
  )
  string(REGEX REPLACE ".* bzip2/libbzip2 version ([0-9]+\\.[^ ]+) of [0-9]+ .*"
                       "\\1" BZIP2_VERSION_STRING "${BZLIB_H}"
  )
endif()

# handle the QUIETLY and REQUIRED arguments and set BZip2_FOUND to TRUE if all
# listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  BZip2
  REQUIRED_VARS BZIP2_LIBRARIES BZIP2_INCLUDE_DIR
  VERSION_VAR BZIP2_VERSION_STRING
)

if(BZIP2_FOUND)
  include(CheckLibraryExists)
  check_library_exists(
    "${BZIP2_LIBRARIES}" BZ2_bzCompressInit "" BZIP2_NEED_PREFIX
  )
endif()

mark_as_advanced(BZIP2_INCLUDE_DIR)
