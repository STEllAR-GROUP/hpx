# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

hpx_find_package(GSL
  LIBRARIES gsl libgsl
  LIBRARY_PATHS lib64 lib
  HEADERS gsl_test.h
  HEADER_PATHS include include/gsl)

if(GSL_FOUND AND NOT HPX_SET_GRL_MACRO)
  set(HPX_SET_GSL_MACRO ON CACHE BOOL "Added the GSL detection macro" FORCE)
  add_definitions(-DGSL_FOUND)
endif()

