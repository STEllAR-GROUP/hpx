# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (vertex_list_HEADERS
    vertex_list.hpp
    server/vertex_list.hpp
    stubs/vertex_list.hpp)

set (vertex_list_SOURCES
    vertex_list.cpp
    server/vertex_list.cpp)

set(vertex_list_DEPENDENCIES
    distributing_factory_component)
    
###############################################################################

set(vertex_list_INI
    vertex_list.ini)
source_group(""
    FILES ${vertex_list_INI})

###############################################################################

add_hpx_component(vertex_list 
    SOURCES ${vertex_list_SOURCES} 
    HEADERS ${vertex_list_HEADERS}
    DEPENDENCIES ${vertex_list_DEPENDENCIES}
    INI ${vertex_list_INI})
