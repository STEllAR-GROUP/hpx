# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(component_EXAMPLES 
    amr)  # amr_client example

foreach(example ${component_EXAMPLES})
    # define build target
    set(${example}_client_SOURCES 
        ${example}_client.cpp)

    add_executable(${example}_client_exe 
        ${${example}_client_SOURCES})

    # avoid conflicts between source and binary target names
    set_target_properties(${example}_client_exe PROPERTIES
        OUTPUT_NAME ${example}_client${CMAKE_DEBUG_POSTFIX})

    # linker instructions
    target_link_libraries(
        ${example}_client_exe                 # executable
        ${example}_component                  # components it depends on
        distributing_factory_component
        amr_component
        amr_test_component
        ${hpx_LIBRARIES} ${Boost_LIBRARIES})  # libraries it dependson

    # installation instructions
    install(TARGETS ${example}_client_exe
        RUNTIME DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)
endforeach()
