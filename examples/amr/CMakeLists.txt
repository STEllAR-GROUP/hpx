# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs 
    amr
    amr_c
    amr_c_test) 

foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()

set(sources
    amr_client.cpp)

source_group("Source Files" FILES ${sources})

# add example executable
add_hpx_executable(amr_client 
  MODULE amr
  SOURCES amr_client.cpp
  DEPENDENCIES amr_component
               amr_c_component
               amr_c_test_component
               distributing_factory_component)
 
# add a dependency to the pseudo-target  
add_hpx_pseudo_dependencies(examples.amr amr_client_exe) 

