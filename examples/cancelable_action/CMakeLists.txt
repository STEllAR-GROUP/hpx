# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Add component for this example
add_subdirectory(cancelable_action)

# Add example itself
set(sources cancelable_action_client.cpp)

source_group("Source Files" FILES ${sources})

# add example executable
add_hpx_executable(cancelable_action_client
    SOURCES ${sources}
    DEPENDENCIES cancelable_action_component
    FOLDER "Examples/CancelableAction")

# add a custom target for this example
#add_hpx_pseudo_target(examples.cancelable_action.cancelable_action)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples examples.cancelable_action)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.cancelable_action
    cancelable_action_client_exe)

