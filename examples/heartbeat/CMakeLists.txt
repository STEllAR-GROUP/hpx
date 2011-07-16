# Copyright (c) 2009-2010 Dylan Stark
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# define build targets for this directory
add_hpx_executable(heartbeat
                   MODULE heartbeat 
                   SOURCES heartbeat.cpp 
                   DEPENDENCIES iostreams_component)

# add pseudo-target dependencies
add_dependencies(examples.heartbeat heartbeat_exe)

