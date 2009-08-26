# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (pbreak_HEADERS
    pbreak.hpp)

set (pbreak_SOURCES
    pbreak.cpp)

###############################################################################

add_hpx_component(pbreak 
    SOURCES ${pbreak_SOURCES} 
    HEADERS ${pbreak_HEADERS})
