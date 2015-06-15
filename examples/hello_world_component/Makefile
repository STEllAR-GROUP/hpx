# Copyright (c) 2012 Steven R. Brandt
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# Linux Bash Script for running hello world

PKG_CONFIG_PATH = $(HPX_LOCATION)/lib/pkgconfig
export PKG_CONFIG_PATH

MY_HPX_LIBS = $(HOME)/my_hpx_libs

# Prepare the environment so that we can run the command
LD_LIBRARY_PATH = $(MY_HPX_LIBS)
export LD_LIBRARY_PATH

all : run

# Compile the library
libhello_world.so : hello_world_component.cpp
	c++ -o libhello_world.so hello_world_component.cpp `pkg-config --cflags --libs hpx_component` -DHPX_COMPONENT_NAME=hello_world

# Move the library to the library directory
$(MY_HPX_LIBS)/libhello_world.so: libhello_world.so
	mkdir -p $(MY_HPX_LIBS)
	cp libhello_world.so $(MY_HPX_LIBS)

# Create the hpx.ini file
$(HOME)/.hpx.ini:
	echo "[hpx]" > $(HOME)/.hpx.ini
	echo "ini_path = \$$[hpx.ini_default_path]:$(MY_HPX_LIBS)" >> $(HOME)/.hpx.ini

# Create the ini file
$(MY_HPX_LIBS)/hello_world.ini:
	echo "$(MY_HPX_LIBS)/hello_world.ini" > $(MY_HPX_LIBS)/hello_world.ini
	echo "[hpx.components.hello_world]" >> $(MY_HPX_LIBS)/hello_world.ini
	echo "name = hello_world" >> $(MY_HPX_LIBS)/hello_world.ini
	echo "path = $(MY_HPX_LIBS)" >> $(MY_HPX_LIBS)/hello_world.ini

# Compile the client
hello_world_client: hello_world_client.cpp $(MY_HPX_LIBS)/libhello_world.so
	c++ -o hello_world_client hello_world_client.cpp `pkg-config --cflags --libs hpx_application` -lhpx_iostreams -lhello_world -L $(MY_HPX_LIBS)

# Run the client, first add our directory to the LD_LIBRARY_PATH
run: hello_world_client $(HOME)/.hpx.ini $(MY_HPX_LIBS)/hello_world.ini
	./hello_world_client

clean:
	rm -f hello_world_client libhello_world.so
