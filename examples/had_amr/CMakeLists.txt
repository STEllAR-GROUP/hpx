# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(had_amr_SUDIRS 
    amr amr_c amr_c_test)

foreach(subdir ${had_amr_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

# define build target for this directory
set(had_amr_client_SOURCES 
    amr_client.cpp)

set(had_amr_DEPENDENCIES
    had_amr_component 
    had_amr_c_component 
    had_amr_c_test_component
    distributing_factory_component)

add_hpx_executable(had_amr_client 
    SOURCES ${had_amr_client_SOURCES}
    DEPENDENCIES ${had_amr_DEPENDENCIES})

