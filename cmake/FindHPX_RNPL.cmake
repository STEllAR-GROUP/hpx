# Copyright (c) 2010 Matt Anderson
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This exports the following macros
# RNPL_FOUND         True if the library and header were found
# RNPL_INCLUDE_DIR   The path to the RNPL header
# RNPL_LIBRARY       The name(s) of the RNPL library to link
# RNPLXX_LIBRARY     The name(s) of the RNPL++ library to link
# RNPL_ROOT          The RNPL main path

################################################################################
# C++-style include guard to prevent multiple searches in the same build
if(NOT RNPL_SEARCHED)
set(RNPL_SEARCHED ON CACHE INTERNAL "Found RNPL library")

if(NOT CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCT)
  set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)
endif()

################################################################################
# Check if RNPL_ROOT is defined and use that path first.
if(NOT RNPL_ROOT AND NOT $ENV{RNPL_ROOT} STREQUAL "")
    set(RNPL_ROOT $ENV{RNPL_ROOT})
endif(NOT RNPL_ROOT AND NOT $ENV{RNPL_ROOT} STREQUAL "")

if(RNPL_ROOT)
    find_path(RNPL_INCLUDE_DIR sdf.h PATHS ${RNPL_ROOT}/include NO_DEFAULT_PATH)
    find_library(RNPL_LIBRARY bbhutil PATHS ${RNPL_ROOT}/lib NO_DEFAULT_PATH)

    if(NOT RNPL_LIBRARY)
        message(WARNING "RNPL not found in ${RNPL_ROOT}.")
        unset(RNPL_ROOT)
    endif(NOT RNPL_LIBRARY)
endif(RNPL_ROOT)

# if not found, retry using system path
if(NOT RNPL_ROOT)
    find_library(RNPL_LIBRARY NAMES bbhutil libbbhutil)
    find_path(RNPL_INCLUDE_DIR sdf.h)
endif(NOT RNPL_ROOT)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RNPL DEFAULT_MSG RNPL_LIBRARY RNPL_INCLUDE_DIR)

if(RNPL_FOUND)
    get_filename_component(RNPL_ROOT ${RNPL_INCLUDE_DIR} PATH)
    set(RNPL_FOUND ${RNPL_FOUND} CACHE BOOL "Found RNPL.")
    set(RNPL_ROOT ${RNPL_ROOT} CACHE PATH "RNPL root directory.")
    set(RNPL_INCLUDE_DIR ${RNPL_INCLUDE_DIR} CACHE PATH "RNPL include directory.")
    set(RNPL_LIBRARY ${RNPL_LIBRARY} CACHE FILEPATH "RNPL shared library.")
endif(RNPL_FOUND)

################################################################################

endif()

