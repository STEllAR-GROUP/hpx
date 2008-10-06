# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (accumulatorlib_HEADERS
    accumulator.hpp
    server/accumulator.hpp
    stubs/accumulator.hpp)

set (accumulatorlib_SOURCES
    accumulator.cpp)

add_definitions(-DHPX_COMPONENT_NAME=accumulator)
add_definitions(-DHPX_COMPONENT_EXPORTS)

add_library (accumulator_component SHARED 
    ${accumulatorlib_SOURCES} ${accumulatorlib_HEADERS})
set_target_properties(accumulator_component PROPERTIES
    OUTPUT_NAME ${component_LIBRARY_PREFIX}accumulator)
target_link_libraries(accumulator_component 
    ${hpx_LIBRARIES} ${Boost_LIBRARIES})

install(TARGETS accumulator_component
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)
