#!/bin/bash
#
#  Copyright (c) 2012 Steven R. Brandt
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# Linux Bash Script for running hello world

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HPX_LOCATION/lib/pkgconfig

# Compile the library
c++ -o libhpx_hello_world.so hello_world_component.cpp `pkg-config --cflags --libs hpx_component` -DHPX_COMPONENT_NAME=hello_world -lhpx_iostreams

# Create the directory where we want to install the library
mkdir -p ~/my_hpx_libs
mv libhpx_hello_world.so ~/my_hpx_libs

# Compile the client
c++ -o hello_world_client hello_world_client.cpp `pkg-config --cflags --libs hpx_application` -L ~/my_hpx_libs -lhpx_hello_world -lhpx_iostreams

# Prepare the environment so that we can run the command
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/my_hpx_libs"

# Run the client, first add our directory to the LD_LIBRARY_PATH
./hello_world_client

