# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Vinay C Amatya
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
# just recurse into all components subdirectories and execute the scripts there

add_subdirectory(${subdir})
  
set(sources ${subdir}_client.cpp)
      
source_group("Source Files" FILES ${sources})
  
# add example executable
add_hpx_executable(${subdir}_client
  MODULE nqueen
  SOURCES ${sources}
  DEPENDENCIES ${subdir}_component)
  
set_property(TARGET ${subdir}_component APPEND
  PROPERTY COMPILE_FLAGS
  "-bigobj")
    
# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.nqueen ${subdir}_client_exe)

