# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# accumulator_client example
set(accumulator_SOURCES 
    accumulator_client.cpp)
add_executable(accumulator_client_exe 
    ${accumulator_SOURCES})

# avoid conflicts between source and binary target names
set_target_properties(accumulator_client_exe PROPERTIES
    OUTPUT_NAME accumulator_client)

target_link_libraries(
    accumulator_client_exe                # executable
    accumulator                           # components it depends on
    ${hpx_LIBRARIES} ${Boost_LIBRARIES})  # libraries it dependson


# simple_accumulator_client example
set(simple_accumulator_SOURCES 
    simple_accumulator_client.cpp)
add_executable(simple_accumulator_client_exe 
    ${simple_accumulator_SOURCES})

# avoid conflicts between source and binary target names
set_target_properties(simple_accumulator_client_exe PROPERTIES
    OUTPUT_NAME simple_accumulator_client)

target_link_libraries(
    simple_accumulator_client_exe         # executable
    simple_accumulator                    # components it depends on
    ${hpx_LIBRARIES} ${Boost_LIBRARIES})  # libraries it dependson

