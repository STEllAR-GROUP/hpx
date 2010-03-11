# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This exports the following macros
# GMP_FOUND         True if the library and header were found
# GMP_INCLUDE_DIR   The path to the GMP header
# GMP_LIBRARY       The name(s) of the GMP library to link
# GMPXX_LIBRARY     The name(s) of the GMP++ library to link
# GMP_ROOT          The GMP main path

# Check if GMP_ROOT is defined and use that path first.
if (NOT GMP_ROOT AND NOT $ENV{GMP_ROOT} STREQUAL "")
    set(GMP_ROOT $ENV{GMP_ROOT})
endif(NOT GMP_ROOT AND NOT $ENV{GMP_ROOT} STREQUAL "")

if(GMP_ROOT)
    find_path(GMP_INCLUDE_DIR gmp.h PATHS ${GMP_ROOT}/include NO_DEFAULT_PATH)
    find_library(GMP_LIBRARY gmp PATHS ${GMP_ROOT}/lib NO_DEFAULT_PATH)

    if(NOT GMP_LIBRARY)
        message(STATUS "Warning: GMP not found in the path specified in GMP_ROOT")
        unset(GMP_ROOT)
    endif(NOT GMP_LIBRARY)
endif(GMP_ROOT)

# if not found, retry using system path
if(NOT GMP_ROOT)
    find_library(GMP_LIBRARY NAMES gmp libgmp)
    find_path(GMP_INCLUDE_DIR gmp.h)
endif(NOT GMP_ROOT)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP DEFAULT_MSG GMP_LIBRARY GMP_INCLUDE_DIR)

if (GMP_FOUND)
    get_filename_component(GMP_ROOT ${GMP_INCLUDE_DIR} PATH)
    set(GMP_ROOT ${GMP_ROOT} CACHE PATH "GMP root directory.")
endif(GMP_FOUND)

mark_as_advanced(GMP_INCLUDE_DIR GMP_LIBRARY)
