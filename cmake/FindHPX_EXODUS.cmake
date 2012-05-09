# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

hpx_find_package(EXODUS
  LIBRARIES exoIIv2c libexoIIv2c
  LIBRARY_PATHS lib64 lib
  HEADERS exodusII.h
  HEADER_PATHS include cbind/include)

if($ENV{EXODUS_ROOT})
  set(EXODUS_FORTRAN_ROOT $ENV{EXODUS_ROOT})
endif()

if(EXODUS_ROOT)
  set(EXODUS_FORTRAN_ROOT ${EXODUS_ROOT})
endif()


hpx_find_package(EXODUS_FORTRAN
  LIBRARIES exoIIv2for libexoIIv2for
  LIBRARY_PATHS lib64 lib
  HEADERS exodusII.inc
  HEADER_PATHS include forbind/include)
