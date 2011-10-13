# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()
    
if(NOT LORENE_ROOT AND NOT $ENV{LORENE_ROOT} STREQUAL "")
  set(LORENE_ROOT $ENV{SDF_ROOT})
endif()

if(HOME_LORENE)
  set(LORENE_ROOT "${HOME_LORENE}")
endif()
 
hpx_find_package(LORENE
  LIBRARIES lorene_export lorene lorenef77
  LIBRARY_PATHS liblib
  HEADERS etoile.h eos.h nbr_spx.h unites.h metric.h type_parite.h proto.h param.h coord.h cmp.h tenseur.h bhole.h bin_ns_bh.h et_rot_mag.h
  HEADER_PATHS C++/Include)

if(LORENE_FOUND)
  add_definitions(-DLORENE_FOUND)
endif()

