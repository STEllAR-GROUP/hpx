# Copyright (c) 2007-2013 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Add all dependencies (and subdirectories) to this target

add_subdirectory(hello_world_component)

# This directory contains the sources for an HPX executable. The following 
# directives add this executable to the build process.

set(example
    hello_world_client)

set(sources
    hello_world_client.cpp)
set(dependencies
    hello_world)

add_hpx_executable(${example}
    SOURCES ${sources}
    COMPONENT_DEPENDENCIES ${dependencies}
    FOLDER "Examples/HelloWorldExternal")
