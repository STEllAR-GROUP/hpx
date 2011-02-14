# Copyright (c) 2010 Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This sets the following variables:
# PXACCEL_ROOT       Root of the acceleration framework install directory
# PXACCEL_FOUND      True if the library and header were found
# PXACCEL_INC_DIR    The path to the public header files
# PXACCEL_LIB_DIR    The path to the library
# PXACCEL_BIN_DIR    The path to the binaries

# Check for explicit definition of root installation directory
if(NOT PXACCEL_ROOT AND NOT $ENV{PXACCEL_ROOT} STREQUAL "")
    set(PXACCEL_ROOT $ENV{PXACCEL_ROOT})
endif()

if(PXACCEL_ROOT)
    find_path(PXACCEL_INC_DIR pci.hh PATHS ${PXACCEL_ROOT}/include NO_DEFAULT_PATH)
    find_library(PXACCEL_LIB pciutil PATHS ${PXACCEL_ROOT}/lib NO_DEFAULT_PATH)
    find_program(PXACCEL_BIN pciconf PATHS ${PXACCEL_ROOT}/bin NO_DEFAULT_PATH)

    if(NOT PXACCEL_LIB OR NOT PXACCEL_INC_DIR)
        message(STATUS "Warning: hardware access support not found in the path specified in PXACCEL_ROOT")
        unset(PXACCEL_ROOT)
    endif()
endif()

# not found; retry with default paths
if(NOT PXACCEL_ROOT)
    find_path(PXACCEL_INC_DIR pci.hh)
    find_library(PXACCEL_LIB pciutil)
    find_program(PXACCEL_BIN pciconf)
endif()

if(PXACCEL_LIB)
    get_filename_component(PXACCEL_LIB_DIR ${PXACCEL_LIB} PATH)
endif()
if(PXACCEL_BIN)
    get_filename_component(PXACCEL_BIN_DIR ${PXACCEL_BIN} PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PXACCEL DEFAULT_MSG PXACCEL_LIB_DIR PXACCEL_INC_DIR PXACCEL_BIN_DIR)

if (PXACCEL_FOUND)
    get_filename_component(PXACCEL_ROOT ${PXACCEL_INC_DIR} PATH)
    set(PXACCEL_ROOT ${PXACCEL_ROOT} CACHE PATH "Root directory of acceleration framework.")
endif()

#mark_as_advanced(PXACCEL_DIR PXACCEL_BIN PXACCEL_LIB PXACCEL_INC_DIR
#                 PXACCEL_LIB_DIR PXACCEL_BIN_DIR)
mark_as_advanced(PXACCEL_ROOT PXACCEL_INC_DIR PXACCEL_LIB_DIR PXACCEL_BIN_DIR)

unset(PXACCEL_BIN CACHE)
unset(PXACCEL_LIB CACHE)
