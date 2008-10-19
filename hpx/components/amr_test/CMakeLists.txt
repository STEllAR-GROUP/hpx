# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(amr_test_HEADERS
    stencil.hpp)

set(amr_test_SOURCES
    amr_test.cpp
    stencil.cpp)

add_definitions(-DHPX_COMPONENT_NAME=amr_test)
add_definitions(-DHPX_COMPONENT_EXPORTS)

add_library (amr_test_component SHARED 
    ${amr_test_SOURCES} ${amr_test_HEADERS})
set_target_properties(amr_test_component PROPERTIES
    OUTPUT_NAME ${component_LIBRARY_PREFIX}amr_test)
target_link_libraries(amr_test_component 
    ${hpx_LIBRARIES} ${Boost_LIBRARIES})

# install binary
install(TARGETS amr_test_component
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

# install hpx configuration (ini) file
install(FILES amr_test.ini
    DESTINATION share/hpx/ini)

