# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# LCO tests
set(barrier_tests_SOURCES barrier_tests.cpp)
add_executable(barrier_tests_exe ${barrier_tests_SOURCES})
set_target_properties(barrier_tests_exe PROPERTIES OUTPUT_NAME barrier_tests)
target_link_libraries(barrier_tests_exe ${hpx_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS barrier_tests_exe
    RUNTIME DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

set(full_empty_tests_SOURCES full_empty_tests.cpp)
add_executable(full_empty_tests_exe ${full_empty_tests_SOURCES})
set_target_properties(full_empty_tests_exe PROPERTIES OUTPUT_NAME full_empty_tests)
target_link_libraries(full_empty_tests_exe ${hpx_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS full_empty_tests_exe
    RUNTIME DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

