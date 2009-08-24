# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (props_HEADERS
    props.hpp
    server/props.hpp
    stubs/props.hpp)

set (props_SOURCES
    props.cpp)

###############################################################################

set(props_INI
    props.ini)
source_group(""
    FILES ${props_INI})

###############################################################################

add_hpx_component(props 
    SOURCES ${props_SOURCES} 
    HEADERS ${props_HEADERS}
    INI ${props_INI})
