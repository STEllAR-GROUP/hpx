# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Component specific tests
set(memory_tests_SOURCES memory_tests.cpp)
add_executable(memory_tests_exe ${memory_tests_SOURCES})
set_target_properties(memory_tests_exe PROPERTIES OUTPUT_NAME memory_tests)
target_link_libraries(memory_tests_exe ${hpx_LIBRARIES} ${Boost_LIBRARIES})
install(TARGETS memory_tests_exe
    RUNTIME DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)
