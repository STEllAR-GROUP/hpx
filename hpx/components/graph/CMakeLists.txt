# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (graph_HEADERS
    graph.hpp
    vertex.hpp
    edge.hpp
    server/graph.hpp
    server/edge.hpp
    stubs/graph.hpp
    stubs/edge.hpp)

set (graph_SOURCES
    graph.cpp
    vertex.cpp
    edge.cpp)
    
set (graph_DEPENDENCIES
    distributing_factory_component
    distributed_set_component)

###############################################################################

set(graph_INI
    graph.ini)
source_group(""
    FILES ${graph_INI})

###############################################################################

add_hpx_component(graph 
    SOURCES ${graph_SOURCES} 
    HEADERS ${graph_HEADERS}
    DEPENDENCIES ${graph_DEPENDENCIES}
    INI ${graph_INI})
