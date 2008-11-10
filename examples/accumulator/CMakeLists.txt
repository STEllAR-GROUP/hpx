# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# define list of target names
set(component_EXAMPLES 
    accumulator          # accumulator_client example
    simple_accumulator)  # simple_accumulator_client example

# for all targets specified above
foreach(example ${component_EXAMPLES})
    # define build target
    set(${example}_client_SOURCES 
        ${example}_client.cpp)

    # add example executable
    add_hpx_executable(${example}_client 
        SOURCES ${${example}_client_SOURCES}
        DEPENDENCIES ${example}_component)
endforeach()
