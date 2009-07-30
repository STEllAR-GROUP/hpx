# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (graph_HEADERS
    graph.hpp
    server/graph.hpp
    stubs/graph.hpp)

set (graph_SOURCES
    graph.cpp)

###############################################################################

set(graph_INI
    graph.ini)
source_group(""
    FILES ${graph_INI})

###############################################################################

add_hpx_component(graph 
    SOURCES ${graph_SOURCES} 
    HEADERS ${graph_HEADERS})
