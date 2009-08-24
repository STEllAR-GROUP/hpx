# Copyright (c) 2009-2010 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (distributed_map_HEADERS
    distributed_map.hpp
    server/distributed_map.hpp
    stubs/distributed_map.hpp
    local_map.hpp
    server/local_map.hpp
    stubs/local_map.hpp)

set (distributed_map_SOURCES
    distributed_map.cpp
    local_map.cpp
    server/distributed_map.cpp
    server/local_map.cpp)

set(distributed_map_DEPENDENCIES
    )

###############################################################################

set(distributed_map_INI
    distributed_map.ini)
source_group(""
    FILES ${distributed_map_INI})

###############################################################################

add_hpx_component(distributed_map 
    SOURCES ${distributed_map_SOURCES} 
    HEADERS ${distributed_map_HEADERS}
    DEPENDENCIES ${distributed_map_DEPENDENCIES}
    INI ${distributed_map_INI})
