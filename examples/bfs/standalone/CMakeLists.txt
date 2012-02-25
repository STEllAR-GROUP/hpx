# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(sources
    bfsc.cpp
    validate.cpp)

source_group("Source Files" FILES ${sources})

# add example executable
add_hpx_executable(bfsc
    MODULE bfsc
    SOURCES ${sources}
    FOLDER "Examples/Bfs Standalone")

# add a custom target for this example
add_hpx_pseudo_target(examples.bfs.standalone.bfsc)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.bfs.standalone examples.bfs.standalone.bfsc)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.bfs.standalone.bfsc bfsc)

