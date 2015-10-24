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

# --------- DOWNLOAD AND BUILD THE EXTERNAL PROJECT! ------------ #
if(NOT ACTIVEHARMONY_FOUND)
  message("Attention: Downloading and Building ActiveHarmony as external project!")
  message(INFO " A working internet connection is required!")
  include(ExternalProject)
  ExternalProject_Add(project_activeharmony
    URL http://www.dyninst.org/sites/default/files/downloads/harmony/ah-4.5.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/activeharmony-4.5
    CONFIGURE_COMMAND ""
    BUILD_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/activeharmony-4.5/src/project_activeharmony && make CFLAGS=-O3
    INSTALL_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/activeharmony-4.5/src/project_activeharmony && make install PREFIX=${CMAKE_CURRENT_BINARY_DIR}/activeharmony-4.5
    ##INSTALL_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/activeharmony-4.5/src/project_activeharmony && make install PREFIX=${CMAKE_INSTALL_PREFIX}
  )
  set(ACTIVEHARMONY_ROOT "${CMAKE_CURRENT_BINARY_DIR}/activeharmony-4.5" CACHE STRING "ActiveHarmony installation directory" FORCE)
  #set(ACTIVEHARMONY_ROOT ${CMAKE_INSTALL_PREFIX})
  ExternalProject_Get_Property(project_activeharmony install_dir)
  add_library(harmony STATIC IMPORTED)
  set_property(TARGET harmony PROPERTY IMPORTED_LOCATION ${install_dir}/lib/libharmony.a)
  set(ACTIVEHARMONY_INCLUDE_DIR "${ACTIVEHARMONY_ROOT}/include")
  set(ACTIVEHARMONY_LIBRARY "${ACTIVEHARMONY_ROOT}/lib/libharmony.a")
  # handle the QUIETLY and REQUIRED arguments and set ACTIVEHARMONY_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(ACTIVEHARMONY  DEFAULT_MSG
                                    ACTIVEHARMONY_LIBRARY ACTIVEHARMONY_INCLUDE_DIR)
  set(ACTIVEHARMONY_FOUND TRUE)
else()
  add_hpx_pseudo_target(project_activeharmony)
endif()

if(ACTIVEHARMONY_FOUND)
  set(ACTIVEHARMONY_LIBRARIES ${ACTIVEHARMONY_LIBRARY} )
  set(ACTIVEHARMONY_INCLUDE_DIRS ${ACTIVEHARMONY_INCLUDE_DIR})
  set(ACTIVEHARMONY_DIR ${ACTIVEHARMONY_ROOT})
  add_definitions(-DAPEX_HAVE_ACTIVEHARMONY)
endif()

