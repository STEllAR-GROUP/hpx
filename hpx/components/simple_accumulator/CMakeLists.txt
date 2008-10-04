# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (simple_accumulatorlib_HEADERS
    simple_accumulator.hpp
    server/simple_accumulator.hpp
    stubs/simple_accumulator.hpp)

set (simple_accumulatorlib_SOURCES
    simple_accumulator.cpp)

add_definitions(-DHPX_COMPONENT_NAME=simple_accumulator)
add_definitions(-DHPX_COMPONENT_EXPORTS)

add_library (simple_accumulator_component SHARED 
    ${simple_accumulatorlib_SOURCES} ${simple_accumulatorlib_HEADERS})
set_target_properties(simple_accumulator_component PROPERTIES
    OUTPUT_NAME ${component_LIBRARY_PREFIX}simple_accumulator)
target_link_libraries(simple_accumulator_component 
    ${hpx_LIBRARIES} ${Boost_LIBRARIES})

install(TARGETS simple_accumulator_component
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)
