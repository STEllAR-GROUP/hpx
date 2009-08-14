# Copyright (c) 2009-2010 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (distributed_list_HEADERS
    distributed_list.hpp
    server/distributed_list.hpp
    stubs/distributed_list.hpp
    local_list.hpp
    server/local_list.hpp
    stubs/local_list.hpp)

set (distributed_list_SOURCES
    distributed_list.cpp
    server/distributed_list.cpp
    server/local_list.cpp)

set(distributed_list_DEPENDENCIES
    )

###############################################################################

set(distributed_list_INI
    distributed_list.ini)
source_group(""
    FILES ${distributed_list_INI})

###############################################################################

add_hpx_component(distributed_list 
    SOURCES ${distributed_list_SOURCES} 
    HEADERS ${distributed_list_HEADERS}
    DEPENDENCIES ${distributed_list_DEPENDENCIES}
    INI ${distributed_list_INI})
