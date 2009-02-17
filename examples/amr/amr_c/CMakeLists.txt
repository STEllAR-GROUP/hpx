# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(amr_c_HEADERS
    logging.hpp
    stencil.hpp
    stencil_data.hpp
    stencil_functions.hpp)

set(amr_c_SOURCES
    amr_c.cpp
    logging.cpp
    stencil.cpp)

set(amr_c_INI
    amr_c.ini)

add_hpx_component(amr_c
    SOURCES ${amr_c_SOURCES} 
    HEADERS ${amr_c_HEADERS}
    INI ${amr_c_INI}
    DEPENDENCIES amr_c_test_component)

