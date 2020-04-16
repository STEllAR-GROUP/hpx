# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Matt Anderson
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Set LORENE_ROOT in case the other hints are used
if(LORENE_ROOT)
  # The call to file is for compatibility with windows paths
  file(TO_CMAKE_PATH ${LORENE_ROOT} LORENE_ROOT)
elseif("$ENV{HOME_LORENE}")
  # This if statement is specific to Lorene, and should not be copied into other
  # Find cmake scripts.
  file(TO_CMAKE_PATH $ENV{HOME_LORENE} LORENE_ROOT)
endif()

if(LORENE_USE_SYSTEM)
  set(LORENE_F77_USE_SYSTEM ON)
endif()

if(LORENE_ROOT)
  set(LORENE_F77_ROOT "${LORENE_ROOT}")
endif()

if(LORENE_USE_SYSTEM)
  set(LORENE_EXPORT_USE_SYSTEM ON)
endif()

if(LORENE_ROOT)
  set(LORENE_EXPORT_ROOT "${LORENE_ROOT}")
endif()

hpx_find_package(LORENE
  LIBRARIES lorene liblorene
  LIBRARY_PATHS lib64 lib Lib
  HEADERS bin_bhns_extr.h
  HEADER_PATHS include include/C++/Include C++/Include)

hpx_find_package(LORENE_F77
  LIBRARIES lorenef77 liblorenef77
  LIBRARY_PATHS lib64 lib Lib
  HEADERS unites.h
  HEADER_PATHS include include/C++/Include C++/Include)

hpx_find_package(LORENE_EXPORT
  LIBRARIES lorene_export liblorene_export
  LIBRARY_PATHS lib64 lib Lib
  HEADERS unites.h
  HEADER_PATHS include include/C++/Include C++/Include)

