# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

if(NOT SDF_ROOT AND NOT $ENV{SDF_ROOT} STREQUAL "")
  set(SDF_ROOT $ENV{SDF_ROOT})
endif()

if(SDF_ROOT)
  set(RNPL_ROOT "${SDF_ROOT}")
endif()

hpx_find_package(RNPL
  LIBRARIES bbhutil libbbhutil
  LIBRARY_PATHS lib64 lib
  HEADERS sdf.h
  HEADER_PATHS include)

if(RNPL_FOUND AND NOT HPX_SET_RNPL_MACRO)
  hpx_add_definitions(-DSDF_FOUND)
  hpx_add_definitions(-DRNPL_FOUND)
endif()

