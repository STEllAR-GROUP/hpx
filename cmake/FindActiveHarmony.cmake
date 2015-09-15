# Copyright (c)      2015 University of Oregon
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# - Try to find LibACTIVEHARMONY
# Once done this will define
#  ACTIVEHARMONY_FOUND - System has ACTIVEHARMONY
#  ACTIVEHARMONY_INCLUDE_DIRS - The ACTIVEHARMONY include directories
#  ACTIVEHARMONY_LIBRARIES - The libraries needed to use ACTIVEHARMONY
#  ACTIVEHARMONY_DEFINITIONS - Compiler switches required for using ACTIVEHARMONY

if(NOT DEFINED $ACTIVEHARMONY_ROOT)
    if(DEFINED ENV{ACTIVEHARMONY_ROOT})
        # message("   env ACTIVEHARMONY_ROOT is defined as $ENV{ACTIVEHARMONY_ROOT}")
        set(ACTIVEHARMONY_ROOT $ENV{ACTIVEHARMONY_ROOT})
    endif()
endif()

find_path(ACTIVEHARMONY_INCLUDE_DIR NAMES hclient.h
    HINTS ${ACTIVEHARMONY_ROOT}/* $ENV{ACTIVEHARMONY_ROOT}/*)

find_library(ACTIVEHARMONY_LIBRARY NAMES harmony
    HINTS ${ACTIVEHARMONY_ROOT}/* $ENV{ACTIVEHARMONY_ROOT}/*)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set ACTIVEHARMONY_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(ACTIVEHARMONY  DEFAULT_MSG
                                  ACTIVEHARMONY_LIBRARY ACTIVEHARMONY_INCLUDE_DIR)

mark_as_advanced(ACTIVEHARMONY_INCLUDE_DIR ACTIVEHARMONY_LIBRARY)

if(ACTIVEHARMONY_FOUND)
  set(ACTIVEHARMONY_LIBRARIES ${ACTIVEHARMONY_LIBRARY} )
  set(ACTIVEHARMONY_INCLUDE_DIRS ${ACTIVEHARMONY_INCLUDE_DIR})
  set(ACTIVEHARMONY_DIR ${ACTIVEHARMONY_ROOT})
  add_definitions(-DAPEX_HAVE_ACTIVEHARMONY)
endif()

