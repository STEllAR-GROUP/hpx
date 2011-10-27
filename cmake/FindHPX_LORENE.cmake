# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

if(NOT LORENE_ROOT AND NOT $ENV{HOME_LORENE} STREQUAL "")
  set(LORENE_ROOT $ENV{HOME_LORENE})
endif()

if(LORENE_ROOT)
  set(LORENE_F77_ROOT $ENV{HOME_LORENE})
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

if(LORENE_FOUND AND NOT HPX_SET_LORENE_MACRO)
  set(HPX_SET_LORENE_MACRO ON CACHE BOOL "Added the Lorene detection macro" FORCE)
  add_definitions(-DLORENE_FOUND)
endif()

