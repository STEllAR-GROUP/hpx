# Copyright (c) 2007-2010 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(refcnt_SUDIRS 
    components)

foreach(subdir ${refcnt_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

# define list of target names for this directory
set(component_EXAMPLES 
    refcnt)

# for all targets specified above
foreach(example ${component_EXAMPLES})
    # define build target
    set(${example}_SOURCES 
        ${example}.cpp)

    # add example executable
    add_hpx_executable(${example}
        SOURCES ${${example}_SOURCES}
        DEPENDENCIES ${example}_component)
endforeach()
