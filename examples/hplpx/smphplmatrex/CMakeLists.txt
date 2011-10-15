# Copyright (C) 2011 Dan Kogler
# Copyright (C) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(smphplmatrex_HEADERS
    smphplmatrex.hpp
    server/smphplmatrex.hpp
    stubs/smphplmatrex.hpp)

set(smphplmatrex_SOURCES
    smphplmatrex.cpp)

add_hpx_component(smphplmatrex 
    SOURCES ${smphplmatrex_SOURCES} 
    HEADERS ${smphplmatrex_HEADERS})

