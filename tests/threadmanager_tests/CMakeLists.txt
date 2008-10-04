# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Thread Manager tests
set(thread_state_test_SOURCES thread_state_test.cpp)
add_executable(thread_state_test_exe ${thread_state_test_SOURCES})
set_target_properties(thread_state_test_exe PROPERTIES OUTPUT_NAME thread_state_test)
target_link_libraries(thread_state_test_exe ${hpx_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS thread_state_test_exe
    RUNTIME DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

set(timed_set_state_tests_SOURCES timed_set_state_tests.cpp)
add_executable(timed_set_state_tests_exe ${timed_set_state_tests_SOURCES})
set_target_properties(timed_set_state_tests_exe PROPERTIES OUTPUT_NAME timed_set_state_tests)
target_link_libraries(timed_set_state_tests_exe ${hpx_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS timed_set_state_tests_exe
    RUNTIME DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

