# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (vertex_HEADERS
    vertex.hpp
    server/vertex.hpp
    stubs/vertex.hpp)

set (vertex_SOURCES
    vertex.cpp)

###############################################################################

set(vertex_INI
    vertex.ini)
source_group(""
    FILES ${vertex_INI})

###############################################################################

add_hpx_component(vertex 
    SOURCES ${vertex_SOURCES} 
    HEADERS ${vertex_HEADERS}
    INI ${vertex_INI})
