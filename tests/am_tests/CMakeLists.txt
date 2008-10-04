# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Action Manager tests 
set(am_test_client_SOURCES am_test_client.cpp)
add_executable(am_test_client_exe ${am_test_client_SOURCES})
set_target_properties(am_test_client_exe PROPERTIES OUTPUT_NAME am_test_client)
target_link_libraries(
    am_test_client_exe                    # executable
    accumulator_component                 # components it depends on
    ${hpx_LIBRARIES} ${Boost_LIBRARIES})  # libraries it dependson
install(TARGETS am_test_client_exe
    RUNTIME DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

set(am_test_server_SOURCES am_test_server.cpp)
add_executable(am_test_server_exe ${am_test_server_SOURCES})
set_target_properties(am_test_server_exe PROPERTIES OUTPUT_NAME am_test_server)
target_link_libraries(
    am_test_server_exe                    # executable
    accumulator_component                 # components it depends on
    ${hpx_LIBRARIES} ${Boost_LIBRARIES})  # libraries it dependson
install(TARGETS am_test_server_exe
    RUNTIME DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

