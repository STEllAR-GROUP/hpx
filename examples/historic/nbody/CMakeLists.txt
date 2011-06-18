# Copyright (c) 2007-2010 Hartmut Kaiser
# Copyright (c) 2009-2010 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs 
    nbody
    nbody_c
    nbody_c_test)


foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()

# define build target for this directory
set(sources 
    nbody_client.cpp)

# define basic dependencies
set(dependencies
    nbody_component 
    nbody_c_component 
    nbody_c_test_lib
    distributing_factory_component)

source_group("Source Files" FILES ${sources})

add_hpx_executable(nbody_client 
  MODULE nbody
  SOURCES ${sources}
  DEPENDENCIES ${dependencies})

# add a dependency to the pseudo-target  
add_hpx_pseudo_dependencies(examples.nbody nbody_client_exe) 

