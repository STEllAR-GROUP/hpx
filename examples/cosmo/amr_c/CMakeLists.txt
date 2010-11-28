# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(cosmo_amr_c_HEADERS
    logging.hpp
    stencil.hpp
    stencil_data.hpp
    stencil_data_locking.hpp
    stencil_functions.hpp)

set(cosmo_amr_c_SOURCES
    amr_c.cpp
    logging.cpp
    stencil.cpp)

set(cosmo_amr_c_INI
    cosmo_amr_c.ini)

add_hpx_component(cosmo_amr_c
    SOURCES ${cosmo_amr_c_SOURCES} 
    HEADERS ${cosmo_amr_c_HEADERS}
    INI ${cosmo_amr_c_INI}
    DEPENDENCIES cosmo_amr_c_test_component)

