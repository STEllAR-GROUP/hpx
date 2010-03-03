# Copyright (c) 2007-2010 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (refcnt_HEADERS
    refcnt.hpp
    server/refcnt.hpp
    stubs/refcnt.hpp)

set (refcnt_SOURCES
     refcnt.cpp)

add_hpx_component(refcnt 
    SOURCES ${refcnt_SOURCES} 
    HEADERS ${refcnt_HEADERS})

