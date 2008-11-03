# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(amr_HEADERS
    stencil_value.hpp
    stubs/stencil_value.hpp
    server/functional_component.hpp
#    server/functional_timestep.hpp
#    server/functional_timestep_impl.hpp
    server/stencil_value.hpp
    server/stencil_value_impl.hpp
    server/stencil_value_in_adaptor.hpp
    server/stencil_value_out_adaptor.hpp)

set(amr_SOURCES
    amr.cpp
    server/functional_component.cpp
#    server/functional_timestep.cpp
    server/stencil_value.cpp
    server/stencil_value_out_adaptor.cpp)

add_definitions(-DHPX_COMPONENT_NAME=amr)
add_definitions(-DHPX_COMPONENT_EXPORTS)

add_library (amr_component SHARED 
    ${amr_SOURCES} ${amr_HEADERS})
set_target_properties(amr_component PROPERTIES
    OUTPUT_NAME ${component_LIBRARY_PREFIX}amr)
target_link_libraries(amr_component 
    ${hpx_LIBRARIES} ${Boost_LIBRARIES})

# install binary
install(TARGETS amr_component
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

# install hpx configuration (ini) file
install(FILES amr.ini
    DESTINATION share/hpx/ini)

