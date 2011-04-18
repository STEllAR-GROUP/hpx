# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
# just recurse into all components subdirectories and execute the scripts there

set(subdirs nqueen)
    
foreach (subdir ${subdirs})
  add_subdirectory(${subdir})
  
  set(sources
      ${subdir}_client.cpp)
      
  source_group("Source Files" FILES ${sources})
  
  # add example executable
  add_hpx_executable(${subdir}_client
    MODULE nqueen
    SOURCES ${sources}
    DEPENDENCIES ${subdir}_component)
      
  # add custom target
  add_hpx_pseudo_target(examples.nqueen.${subdir})
  
  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.nqueen examples.nqueen.${subdir})
                   
  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.nqueen.${subdir} ${subdir}_client_exe)
                  
endforeach()
