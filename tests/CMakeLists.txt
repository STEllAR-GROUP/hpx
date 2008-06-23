# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (am_test_client_SOURCES am_test_client/am_test_client.cpp)
add_executable (am_test_client ${am_test_client_SOURCES})
target_link_libraries(am_test_client hpx ${Boost_LIBRARIES})

set (am_test_server_SOURCES am_test_server/am_test_server.cpp)
add_executable (am_test_server ${am_test_server_SOURCES})
target_link_libraries(am_test_server hpx ${Boost_LIBRARIES})

set (applier_test_client_SOURCES applier_test_client/applier_test_client.cpp)
add_executable (applier_test_client ${applier_test_client_SOURCES})
target_link_libraries(applier_test_client hpx ${Boost_LIBRARIES})

set (applier_test_server_SOURCES applier_test_server/applier_test_server.cpp)
add_executable (applier_test_server ${applier_test_server_SOURCES})
target_link_libraries(applier_test_server hpx ${Boost_LIBRARIES})

set (thread_state_test_SOURCES thread_state_test/thread_state_test.cpp)
add_executable (thread_state_test ${thread_state_test_SOURCES})
target_link_libraries(thread_state_test hpx ${Boost_LIBRARIES})
