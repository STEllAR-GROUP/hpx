# Copyright (C) 2011 Dan Kogler
# Copyright (C) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(hplmatrex_HEADERS
    hplmatrex.hpp
    server/hplmatrex.hpp
    stubs/hplmatrex.hpp)

set(hplmatrex_SOURCES
    hplmatrex.cpp)

add_hpx_component(hplmatrex
    SOURCES ${hplmatrex_SOURCES}
    HEADERS ${hplmatrex_HEADERS}
    FOLDER "Examples/HplPx")

set(lublock_HEADERS
    lublock.hpp
    server/lublock.hpp
    stubs/lublock.hpp)

set(lublock_SOURCES
    lublock.cpp)

add_hpx_component(lublock
    SOURCES ${lublock_SOURCES}
    HEADERS ${lublock_HEADERS}
    FOLDER "Examples/HplPx")

