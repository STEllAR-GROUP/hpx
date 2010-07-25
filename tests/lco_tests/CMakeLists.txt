# Copyright (c) 2007-2010 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# LCO tests
set(TESTS 
    #barrier_tests 
    #local_barrier_tests 
    #full_empty_tests 
    #counting_semaphore_test
    #mutex_tests
    #lazy_future_tests
    #thunk_tests
    #dataflow_variable_tests
   )

# build and install each of the tests
foreach(test ${TESTS})
    set(${test}_SOURCES ${test}.cpp)
    add_hpx_executable(${test} SOURCES ${${test}_SOURCES})
endforeach()

