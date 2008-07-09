# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Single file tests
set (am_test_client_SOURCES am_test_client/am_test_client.cpp)
add_executable (am_test_client_exe ${am_test_client_SOURCES})
target_link_libraries(am_test_client_exe hpx ${Boost_LIBRARIES})

set (am_test_server_SOURCES am_test_server/am_test_server.cpp)
add_executable (am_test_server_exe ${am_test_server_SOURCES})
target_link_libraries(am_test_server_exe hpx ${Boost_LIBRARIES})

set (applier_test_client_SOURCES applier_test_client/applier_test_client.cpp)
add_executable (applier_test_client_exe ${applier_test_client_SOURCES})
target_link_libraries(applier_test_client_exe hpx ${Boost_LIBRARIES})

set (applier_test_server_SOURCES applier_test_server/applier_test_server.cpp)
add_executable (applier_test_server_exe ${applier_test_server_SOURCES})
target_link_libraries(applier_test_server_exe hpx ${Boost_LIBRARIES})

set (thread_state_test_SOURCES thread_state_test/thread_state_test.cpp)
add_executable (thread_state_test_exe ${thread_state_test_SOURCES})
target_link_libraries(thread_state_test_exe hpx ${Boost_LIBRARIES})

set (barrier_tests_SOURCES lco_tests/barrier_tests.cpp)
add_executable (barrier_tests_exe ${barrier_tests_SOURCES})
target_link_libraries(barrier_tests_exe hpx ${Boost_LIBRARIES})

set (memory_tests_SOURCES memory_tests/memory_tests.cpp)
add_executable (memory_tests_exe ${memory_tests_SOURCES})
target_link_libraries(memory_tests_exe hpx ${Boost_LIBRARIES})

# performance/measurement tests
set (measure_thread_execution_SOURCES performance/measure_thread_execution.cpp)
add_executable (measure_thread_execution_exe ${measure_thread_execution_SOURCES})
target_link_libraries(measure_thread_execution_exe hpx ${Boost_LIBRARIES})
