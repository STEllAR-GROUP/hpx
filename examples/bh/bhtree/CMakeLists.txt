# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (bhtree_HEADERS
    bhtree.hpp
    server/bhtree.hpp
    stubs/bhtree.hpp)

set (bhtree_SOURCES
    bhtree.cpp)

add_hpx_component(IntrTreeNode
    SOURCES ${bhtree_SOURCES} 
    HEADERS ${bhtree_HEADERS})

