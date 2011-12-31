# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

if(NOT LAPACK_ROOT AND NOT $ENV{HOME_LAPACK} STREQUAL "")
  set(LAPACK_ROOT $ENV{HOME_LAPACK})
endif()

if(LAPACK_USE_SYSTEM)
  set(LAPACK_F77_CPP_USE_SYSTEM ON)
endif()

if(LAPACK_USE_SYSTEM)
  set(LAPACK_EXPORT_CPP_USE_SYSTEM ON)
endif()

hpx_find_package(LAPACK
  LIBRARIES lapack liblapack
  LIBRARY_PATHS lib64 lib Lib
  HEADERS bbhutil.h
  HEADER_PATHS include include/C++/Include C++/Include)
