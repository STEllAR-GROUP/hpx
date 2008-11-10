# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(component_EXAMPLES 
    amr)  # amr_client example

foreach(example ${component_EXAMPLES})
    # define build target
    set(${example}_client_SOURCES 
        ${example}_client.cpp)

    add_hpx_executable(${example}_client 
        SOURCES ${${example}_client_SOURCES})
endforeach()
