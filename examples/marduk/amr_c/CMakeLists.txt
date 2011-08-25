# Copyright (c) 2009-2011 Matthew Anderson
# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
set(sources
    amr_c.cpp
    logging.cpp
    stencil.cpp)

set(headers
    logging.hpp
    stencil.hpp
    stencil_data.hpp
    stencil_data_locking.hpp
    stencil_functions.hpp)

###############################################################################
source_group("Source Files" FILES ${sources})
source_group("Header Files" FILES ${headers})

set(dependencies
    ssmp_amr3d_c_test_lib
    marduk_mesh_component)

if(RNPL_FOUND)
  # add RNPL as a dependency
  set(dependencies 
      ${dependencies}
      ${RNPL_LIBRARY})
endif()

###############################################################################
add_hpx_component(ssmp_amr3d_c
  MODULE ssmp_amr3d
  SOURCES ${sources} 
  HEADERS ${headers}
  DEPENDENCIES ${dependencies}) 

