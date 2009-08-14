# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (kernel2_HEADERS
    kernel2.hpp
    server/kernel2.hpp
    stubs/kernel2.hpp)

set (kernel2_SOURCES
    kernel2.cpp
    server/kernel2.cpp
    reduce_max.cpp)

set(kernel2_DEPENDENCIES
    graph_component
    distributed_list_component
    distributing_factory_component
    vertex_component)

set(kernel2_INI
    kernel2.ini)
source_group("" 
    FILES ${kernel2_INI})

###############################################################################

add_hpx_component(kernel2 
    SOURCES ${kernel2_SOURCES} 
    HEADERS ${kernel2_HEADERS}
    DEPENDENCIES ${kernel2_DEPENDENCIES}
    INI ${kernel2_INI})
