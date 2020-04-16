# Copyright (c) 2011 Matt Anderson
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

hpx_find_package(EXODUS
  LIBRARIES exoIIv2c libexoIIv2c
  LIBRARY_PATHS lib64 lib
  HEADERS exodusII.h
  HEADER_PATHS include cbind/include)

# Compatibility for windows paths
if(EXODUS_ROOT)
  file(TO_CMAKE_PATH ${EXODUS_ROOT} EXODUS_FORTRAN_ROOT)
elseif("$ENV{EXODUS_ROOT}")
  file(TO_CMAKE_PATH $ENV{EXODUS_ROOT} EXODUS_FORTRAN_ROOT)
endif()

hpx_find_package(EXODUS_FORTRAN
  LIBRARIES exoIIv2for libexoIIv2for
  LIBRARY_PATHS lib64 lib
  HEADERS exodusII.inc
  HEADER_PATHS include forbind/include)
