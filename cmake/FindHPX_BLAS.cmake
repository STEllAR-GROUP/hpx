# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

if(NOT BLAS_ROOT AND NOT $ENV{HOME_BLAS} STREQUAL "")
  set(BLAS_ROOT $ENV{HOME_BLAS})
endif()

if(BLAS_USE_SYSTEM)
  set(BLAS_F77_CPP_USE_SYSTEM ON)
endif()

if(BLAS_USE_SYSTEM)
  set(BLAS_EXPORT_CPP_USE_SYSTEM ON)
endif()

hpx_find_package(BLAS
  LIBRARIES blas libblas
  LIBRARY_PATHS lib64 lib Lib
  HEADERS bbhutil.h
  HEADER_PATHS include include/C++/Include C++/Include)
