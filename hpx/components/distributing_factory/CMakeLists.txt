# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (distributing_factorylib_HEADERS
#    distributing_factory.hpp
#    stubs/distributing_factory.hpp
    server/distributing_factory.hpp)

set (distributing_factorylib_SOURCES
    server/distributing_factory.cpp
    distributing_factory.cpp)

add_definitions(-DHPX_COMPONENT_NAME=distributing_factory)
add_definitions(-DHPX_COMPONENT_EXPORTS)

add_library (distributing_factory_component SHARED 
    ${distributing_factorylib_SOURCES} ${distributing_factorylib_HEADERS})
set_target_properties(distributing_factory_component PROPERTIES
    OUTPUT_NAME ${component_LIBRARY_PREFIX}distributing_factory)
target_link_libraries(distributing_factory_component 
    ${hpx_LIBRARIES} ${Boost_LIBRARIES})
