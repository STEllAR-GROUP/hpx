# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (bh_HEADERS
    bh.hpp
    server/bh.hpp
    stubs/bh.hpp)

set (bh_SOURCES
    bh.cpp)

add_hpx_component(bh 
    SOURCES ${bh_SOURCES} 
    HEADERS ${bh_HEADERS})

