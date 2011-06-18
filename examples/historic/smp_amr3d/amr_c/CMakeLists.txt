# Copyright (c) 2007-2009 Hartmut Kaiser
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

###############################################################################
add_hpx_component(smp_amr3d_c
  MODULE smp_amr3d
  SOURCES ${sources} 
  HEADERS ${headers}
  DEPENDENCIES smp_amr3d_c_test_lib)

