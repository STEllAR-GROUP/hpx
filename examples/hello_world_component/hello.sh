#!/bin/bash
#  Copyright (c) 2012 Steven R. Brandt
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# Linux Bash Script for running hello world

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HPX_LOCATION/lib/pkgconfig

# Compile the library
c++ -o libhello_world.so hello_world_component.cpp `pkg-config --cflags --libs hpx_component` -DHPX_COMPONENT_NAME=hello_world

# Create the directory where we want to install the library
mkdir -p ~/my_hpx_libs
mv libhello_world.so ~/my_hpx_libs

# If we don't have a .hpx.ini yet, create one and tell it about
# our private hpx library directory
if [ ! -r ~/.hpx.ini ]
then
cat > ~/.hpx.ini <<EOF
[hpx]
ini_path = \$[hpx.ini_path]:${HOME}/my_hpx_libs
EOF
fi

# Create the ini file
cat > ~/my_hpx_libs/hello_world.ini <<EOF
[hpx.components.hello_world]
name = hello_world
path = ${HOME}/my_hpx_libs
EOF

# Compile the client
c++ -o hello_world_client hello_world_client.cpp `pkg-config --cflags --libs hpx_application` -lhpx_iostreams -lhello_world -L ~/my_hpx_libs

# Prepare the environment so that we can run the command
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/my_hpx_libs"

# Run the client, first add our directory to the LD_LIBRARY_PATH
./hello_world_client 

