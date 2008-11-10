# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# define build target
set(amr_client_SOURCES 
    amr_client.cpp)

set(amr_DEPENDENCIES
    amr_component 
    amr_test_component 
    distributing_factory_component)

add_hpx_executable(amr_client 
    SOURCES ${amr_client_SOURCES}
    DEPENDENCIES ${amr_DEPENDENCIES})

