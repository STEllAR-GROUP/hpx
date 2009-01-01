# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Action Manager tests 
set(TESTS 
    am_test_client 
    am_test_server)

# build and install each of the tests
foreach(test ${TESTS})
    set(${test}_SOURCES ${test}.cpp)
    add_hpx_executable(${test} 
        SOURCES ${${test}_SOURCES}
        DEPENDENCIES accumulator_component)
endforeach()

