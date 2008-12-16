# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(generic_HEADERS
    generic_component_noresult.hpp
    generic_component_result.hpp)

set(generic_SOURCES
    generic_module.cpp
    generic_component_noresult.cpp
    generic_component_result.cpp)

add_hpx_component(generic 
    SOURCES ${generic_SOURCES} 
    HEADERS ${generic_HEADERS})

