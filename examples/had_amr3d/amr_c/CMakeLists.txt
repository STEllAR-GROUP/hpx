# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(had_amr3d_c_HEADERS
    logging.hpp
    stencil.hpp
    stencil_data.hpp
    stencil_data_locking.hpp
    stencil_functions.hpp)

set(had_amr3d_c_SOURCES
    amr_c.cpp
    logging.cpp
    stencil.cpp)

set(had_amr3d_c_INI
    )

add_hpx_component(had_amr3d_c
    SOURCES ${had_amr3d_c_SOURCES} 
    HEADERS ${had_amr3d_c_HEADERS}
    INI ${had_amr3d_c_INI}
    DEPENDENCIES had_amr3d_c_test_component)

