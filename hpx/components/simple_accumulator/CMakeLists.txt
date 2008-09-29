# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (simple_accumulatorlib_HEADERS
    simple_accumulator.hpp
    server/simple_accumulator.hpp
    stubs/simple_accumulator.hpp
)

set (acumulatorlib_SOURCES
    simple_accumulator.cpp
)

add_definitions(-DHPX_COMPONENT_NAME=simple_accumulator)
add_definitions(-DHPX_COMPONENT_EXPORTS)

add_library (
    simple_accumulator 
    SHARED 
    ${simple_accumulatorlib_SOURCES} 
    ${simple_accumulatorlib_HEADERS}
)

