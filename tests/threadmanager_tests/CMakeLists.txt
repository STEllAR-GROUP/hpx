# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Thread Manager tests
set(TESTS 
    thread_state_test 
    timed_set_state_tests 
    exception_test)

# build and install each of the tests
foreach(test ${TESTS})
    set(${test}_SOURCES ${test}.cpp)
    add_executable(${test}_exe ${${test}_SOURCES})
    set_target_properties(${test}_exe PROPERTIES OUTPUT_NAME ${test})
    target_link_libraries(${test}_exe ${hpx_LIBRARIES} ${BOOST_FOUND_LIBRARIES})
    install(TARGETS ${test}_exe
        RUNTIME DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)
endforeach()

