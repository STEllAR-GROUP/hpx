# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(generic)

set(sources
    generic_client.cpp)

source_group("Source Files" FILES ${sources})

# add example executable
add_hpx_executable(generic_client 
  MODULE generic
  SOURCES ${sources} 
  DEPENDENCIES generic_component)

# add dependencies to pseudo-target 
add_dependencies(examples.generic
                 generic_client 
                 generic_client_exe) 

