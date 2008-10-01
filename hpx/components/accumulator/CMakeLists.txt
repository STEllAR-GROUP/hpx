# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (accumulatorlib_HEADERS
    accumulator.hpp
    server/accumulator.hpp
    stubs/accumulator.hpp
)

set (accumulatorlib_SOURCES
    accumulator.cpp
)

add_definitions(-DHPX_COMPONENT_NAME=accumulator)
add_definitions(-DHPX_COMPONENT_EXPORTS)

add_library (
    accumulator 
    SHARED 
    ${accumulatorlib_SOURCES} 
    ${accumulatorlib_HEADERS}
)

target_link_libraries(accumulator ${hpx_LIBRARIES} ${Boost_LIBRARIES})
