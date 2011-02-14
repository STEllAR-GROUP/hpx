# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(hpx_ACCUMULATOR_SUBDIRS 
    amr
    amr_c
    amr_c_test) 

foreach(hpx_ACCUMULATOR_SUBDIR ${hpx_ACCUMULATOR_SUBDIRS})
  add_subdirectory(${hpx_ACCUMULATOR_SUBDIR})
endforeach()

# define build target
set(amr_client_SOURCES 
    amr_client.cpp)

# add example executable
add_hpx_executable(amr_client 
  SOURCES ${amr_client_SOURCES}
  DEPENDENCIES amr_component
               amr_c_component
               amr_c_test_component
               distributing_factory_component)
 
# add a dependency to the pseudo-target  
add_dependencies(examples.amr amr_client_exe) 

