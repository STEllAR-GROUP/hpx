#  Copyright (c) 2012 Vinay C Amatya
#
#  Distributed under the Boost Software License, Version 1.0. (Seec accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=10)
#add_definitions(-DHPX_COMPONENT_CREATE_ARG_MAX=10)
add_definitions(-DHPX_FUNCTION_LIMIT=13)

# add a custom target for this example
add_hpx_pseudo_target(examples.fft.fft)

add_subdirectory(fft)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.fft examples.fft.fft)
#=======================

source_group("Source Files" FILES fft_client.cpp)

# add example executable
add_hpx_executable(fft_client
    SOURCES fft_client.cpp
    DEPENDENCIES fft_component
    FOLDER "Examples/Fft")

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.fft fft_client_exe)

