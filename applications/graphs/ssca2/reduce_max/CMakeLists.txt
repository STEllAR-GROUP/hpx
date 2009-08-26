# Copyright (c) 2007-2009 Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################

set (reduce_max_HEADERS
    reduce_max.hpp)

set (reduce_max_SOURCES
    reduce_max.cpp)

###############################################################################

add_hpx_component(reduce_max 
    SOURCES ${reduce_max_SOURCES} 
    HEADERS ${reduce_max_HEADERS})
