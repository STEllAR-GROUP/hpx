# Copyright (C) 2011 Daniel Kogler
# Copyright (C) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(bhcontroller_HEADERS
    bhcontroller.hpp
    server/bhcontroller.hpp
    stubs/bhcontroller.hpp)

set(bhcontroller_SOURCES
    bhcontroller.cpp)

add_hpx_component(bhcontroller 
    MODULE bhcontroller
    SOURCES ${bhcontroller_SOURCES} 
    HEADERS ${bhcontroller_HEADERS})

set(bhnode_HEADERS
    bhnode.hpp
    server/bhnode.hpp
    stubs/bhnode.hpp)

set(bhnode_SOURCES
    bhnode.cpp)

add_hpx_component(bhnode 
    MODULE bhcontroller
    SOURCES ${bhnode_SOURCES} 
    HEADERS ${bhnode_HEADERS})

