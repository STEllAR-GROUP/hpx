# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (ssca2_HEADERS
    ssca2.hpp
    server/ssca2.hpp
    stubs/ssca2.hpp)

set (ssca2_SOURCES
    ssca2.cpp
    server/ssca2.cpp
    reduce_max.cpp)

set(ssca2_DEPENDENCIES
    graph_component
    distributed_list_component
    distributing_factory_component
    vertex_component)

set(ssca2_INI
    ssca2.ini)
source_group("" 
    FILES ${ssca2_INI})

###############################################################################

add_hpx_component(ssca2 
    SOURCES ${ssca2_SOURCES} 
    HEADERS ${ssca2_HEADERS}
    DEPENDENCIES ${ssca2_DEPENDENCIES}
    INI ${ssca2_INI})
