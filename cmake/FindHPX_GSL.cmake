# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Matt Anderson
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# This if statement is specific to GSL, and should not be copied into other
# Find cmake scripts.
if(NOT GSL_ROOT AND NOT $ENV{HOME_GSL} STREQUAL "")
  set(GSL_ROOT "$ENV{HOME_GSL}")
endif()

if(GSL_USE_SYSTEM)
  set(GSLCBLAS_USE_SYSTEM ON)
endif()

if(GSL_ROOT)
  set(GSLCBLAS_ROOT "${GSL_ROOT}")
endif()

hpx_find_package(GSL
  LIBRARIES gsl libgsl
  LIBRARY_PATHS lib64 lib
  HEADERS gsl/gsl_test.h
  HEADER_PATHS include)

hpx_find_package(GSLCBLAS
  LIBRARIES gslcblas libgslcblas
  LIBRARY_PATHS lib64 lib
  HEADERS gsl/gsl_blas.h
  HEADER_PATHS include)

