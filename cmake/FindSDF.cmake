# Copyright (c) 2010 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This exports the following macros
# SDF_FOUND         True if the library and header were found
# SDF_INCLUDE_DIR   The path to the SDF header
# SDF_LIBRARY       The name(s) of the SDF library to link
# SDFXX_LIBRARY     The name(s) of the SDF++ library to link
# SDF_ROOT          The SDF main path

# Check if SDF_ROOT is defined and use that path first.
if(NOT SDF_ROOT AND NOT $ENV{SDF_ROOT} STREQUAL "")
    set(SDF_ROOT $ENV{SDF_ROOT})
endif(NOT SDF_ROOT AND NOT $ENV{SDF_ROOT} STREQUAL "")

if(SDF_ROOT)
    find_path(SDF_INCLUDE_DIR sdf.h PATHS ${SDF_ROOT}/include NO_DEFAULT_PATH)
    find_library(SDF_LIBRARY bbhutil PATHS ${SDF_ROOT}/lib NO_DEFAULT_PATH)

    if(NOT SDF_LIBRARY)
        message(STATUS "Warning: SDF not found in the path specified in SDF_ROOT")
        unset(SDF_ROOT)
    endif(NOT SDF_LIBRARY)
endif(SDF_ROOT)

# if not found, retry using system path
if(NOT SDF_ROOT)
    find_library(SDF_LIBRARY NAMES bbhutil libbbhutil)
    find_path(SDF_INCLUDE_DIR sdf.h)
endif(NOT SDF_ROOT)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SDF DEFAULT_MSG SDF_LIBRARY SDF_INCLUDE_DIR)

if (SDF_FOUND)
    get_filename_component(SDF_ROOT ${SDF_INCLUDE_DIR} PATH)
    set(SDF_ROOT ${SDF_ROOT} CACHE PATH "SDF root directory.")
    mark_as_advanced(SDF_INCLUDE_DIR SDF_LIBRARY)
endif(SDF_FOUND)

