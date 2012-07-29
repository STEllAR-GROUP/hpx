#  Copyright (c) 2012 Vinay C Amatya
#
#  Distributed under the Boost Software License, Version 1.0. (Seec accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=10)
add_definitions(-DHPX_FUNCTION_ARGUMENT_LIMIT=13)


# add a custom target for this example
add_hpx_pseudo_target(examples.distributed_test.distributed_test)

add_subdirectory(distributed_test)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.distributed_test examples.distributed_test.distributed_test)
#=======================

source_group("Source Files" FILES distributed_client.cpp)

# add example executable
add_hpx_executable(distributed_client
    SOURCES distributed_client.cpp
    DEPENDENCIES distributed_test_component
    FOLDER "Examples/Distributed Test")

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.distributed_test distributed_client_exe)

