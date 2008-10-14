# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(genericlib_HEADERS
    generic_component_noresult.hpp
    generic_component_result.hpp)

set(genericlib_SOURCES
    generic_module.cpp
    generic_component_noresult.cpp
    generic_component_result.cpp)

add_definitions(-DHPX_COMPONENT_NAME=generic)
add_definitions(-DHPX_COMPONENT_EXPORTS)

add_library (generic_component SHARED 
    ${genericlib_SOURCES} ${genericlib_HEADERS})
set_target_properties(generic_component PROPERTIES
    OUTPUT_NAME ${component_LIBRARY_PREFIX}generic)
target_link_libraries(generic_component 
    ${hpx_LIBRARIES} ${Boost_LIBRARIES})

# install binary
install(TARGETS generic_component
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

# install hpx configuration (ini) file
install(FILES generic.ini
    DESTINATION share/hpx/ini)

