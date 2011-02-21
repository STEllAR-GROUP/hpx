# Copyright (c) 2007-2011 Chirag Dekate
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs 
    tlf
    itn)

foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()

# define build target
set(sources 
    bhc.cpp)

source_group("Source Files" FILES ${sources})

# add example executable
add_hpx_executable(bhc_client
  MODULE bhpx
  SOURCES ${sources}
  DEPENDENCIES tlf_component 
               itn_component
               distributing_factory_component)

# add a dependency to the pseudo-target  
add_dependencies(examples.bhpx
                 bhc_client_exe) 

