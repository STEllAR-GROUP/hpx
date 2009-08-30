# Copyright (c) 2009-2010 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (distributed_set_HEADERS
    distributed_set.hpp
    server/distributed_set.hpp
    stubs/distributed_set.hpp
    local_set.hpp
    server/local_set.hpp
    stubs/local_set.hpp)

set (distributed_set_SOURCES
    distributed_set.cpp
    local_set.cpp
    server/distributed_set.cpp
    server/local_set.cpp)

set(distributed_set_DEPENDENCIES
    distributing_factory_component
    )

###############################################################################

set(distributed_set_INI
    distributed_set.ini)
source_group(""
    FILES ${distributed_set_INI})

###############################################################################

add_hpx_component(distributed_set 
    SOURCES ${distributed_set_SOURCES} 
    HEADERS ${distributed_set_HEADERS}
    DEPENDENCIES ${distributed_set_DEPENDENCIES}
    INI ${distributed_set_INI})
