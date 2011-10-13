# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

if(NOT LORENE_ROOT AND NOT $ENV{LORENE_HOME} STREQUAL "")
  set(LORENE_ROOT $ENV{LORENE_HOME})
endif()
 
hpx_find_package(LORENE
  LIBRARIES lorene_export lorene lorenef77 liblorene_export liblorene liblorenef77
  LIBRARY_PATHS lib64 lib liblib
  HEADERS bin_bhns_extr.h  
  HEADER_PATHS include include/C++/Include C++/Include)

if(LORENE_FOUND AND NOT HPX_SET_LORENE_MACRO)
  set(HPX_SET_LORENE_MACRO ON CACHE BOOL "Added the Lorene detection macro" FORCE)
  add_definitions(-DLORENE_FOUND)
endif()

