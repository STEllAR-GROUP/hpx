# Copyright (c) 2010 Maciej Brodowicz
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This sets the following variables:
# PXACCEL_ROOT       Root of the acceleration framework install directory
# PXACCEL_FOUND      True if the library and header were found
# PXACCEL_INCLUDE_DIR    The path to the public header files
# PXACCEL_LIBRARY_DIR    The path to the library
# PXACCEL_BINARY_DIR    The path to the binaries

################################################################################
# C++-style include guard to prevent multiple searches in the same build
if(NOT PXACCEL_SEARCHED)
set(PXACCEL_SEARCHED ON CACHE INTERNAL "Found hardware acceleration")

if(NOT CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCT)
  set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
endif()

################################################################################
# Check for explicit definition of root installation directory
if(NOT PXACCEL_USE_SYSTEM)
  if(NOT PXACCEL_ROOT AND NOT $ENV{PXACCEL_ROOT} STREQUAL "")
    set(PXACCEL_ROOT $ENV{PXACCEL_ROOT})
  endif()
endif()

if(PXACCEL_ROOT)
    find_path(PXACCEL_INCLUDE_DIR pci.hh PATHS ${PXACCEL_ROOT}/include NO_DEFAULT_PATH)
    find_library(PXACCEL_LIB pciutil PATHS ${PXACCEL_ROOT}/lib NO_DEFAULT_PATH)
    find_program(PXACCEL_BIN pciconf PATHS ${PXACCEL_ROOT}/bin NO_DEFAULT_PATH)

    if(NOT PXACCEL_LIB OR NOT PXACCEL_INCLUDE_DIR)
        message(WARNING "Hardware access support not found in ${PXACCEL_ROOT}.")
        unset(PXACCEL_ROOT)
    endif()
endif()

# not found; retry with default paths
if(NOT PXACCEL_ROOT)
    find_path(PXACCEL_INCLUDE_DIR pci.hh)
    find_library(PXACCEL_LIB pciutil)
    find_program(PXACCEL_BIN pciconf)
endif()

if(PXACCEL_LIB)
    get_filename_component(PXACCEL_LIBRARY_DIR ${PXACCEL_LIB} PATH)
endif()

if(PXACCEL_BIN)
    get_filename_component(PXACCEL_BINARY_DIR ${PXACCEL_BIN} PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PXACCEL DEFAULT_MSG PXACCEL_LIBRARY_DIR PXACCEL_INCLUDE_DIR PXACCEL_BINARY_DIR)

if(PXACCEL_FOUND)
    get_filename_component(PXACCEL_ROOT ${PXACCEL_INCLUDE_DIR} PATH)
    set(PXACCEL_FOUND ${PXACCEL_FOUND} CACHE BOOL "Found PXACCEL.")
    set(PXACCEL_ROOT ${PXACCEL_ROOT} CACHE PATH "Root directory of hardware acceleration framework.")
    set(PXACCEL_INCLUDE_DIR ${PXACCEL_INCLUDE_DIR} CACHE PATH "PXACCEL include directory.")
    set(PXACCEL_LIBRARY_DIR ${PXACCEL_LIBRARY_DIR} CACHE PATH "PXACCEL shared library directory.")
    set(PXACCEL_LIB ${PXACCEL_LIB} CACHE FILEPATH "PXACCEL shared library.")
    set(PXACCEL_BINARY_DIR ${PXACCEL_BINARY_DIR} CACHE PATH "PXACCEL binary directory.")
endif()

################################################################################

endif()

