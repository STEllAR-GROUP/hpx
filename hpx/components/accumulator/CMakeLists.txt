# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (accumulator_HEADERS
    accumulator.hpp
    server/accumulator.hpp
    stubs/accumulator.hpp)

set (accumulator_SOURCES
    accumulator.cpp)

add_hpx_component(accumulator 
    SOURCES ${accumulator_SOURCES} 
    HEADERS ${accumulator_HEADERS})

