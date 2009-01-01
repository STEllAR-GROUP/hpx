# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(generic_SUDIRS 
    generic)

foreach(subdir ${generic_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

# define list of target names for this directory
set(component_EXAMPLES 
    generic)              # example demonstrating generic_component

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

