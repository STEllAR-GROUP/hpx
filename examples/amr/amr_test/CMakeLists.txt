# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(amr_test_HEADERS
    logging.hpp
    stencil.hpp
    stencil_data.hpp)

set(amr_test_SOURCES
    amr_test.cpp
    logging.cpp
    stencil.cpp)

set(amr_test_INI
    amr_test.ini)

add_hpx_component(amr_test 
    SOURCES ${amr_test_SOURCES} 
    HEADERS ${amr_test_HEADERS}
    INI ${amr_test_INI})

