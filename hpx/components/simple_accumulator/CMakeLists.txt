# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (simple_accumulator_HEADERS
    simple_accumulator.hpp
    server/simple_accumulator.hpp
    stubs/simple_accumulator.hpp)

set (simple_accumulator_SOURCES
    simple_accumulator.cpp)

add_hpx_component(simple_accumulator 
    SOURCES ${simple_accumulator_SOURCES} 
    HEADERS ${simple_accumulator_HEADERS})

