# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2007-2010 Hartmut Kaiser
# Copyright (c) 2009-2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs 
    dataflow
    stencil
    application)

foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()

# define build target for this directory
set(sources
    adaptive1d.cpp)

# define basic dependencies
set(dependencies
    adaptive1d_dataflow_component)

source_group("Source Files" FILES ${sources})

set(hpx_LIBRARIES ${hpx_LIBRARIES} gfortran)

add_hpx_executable(adaptive1d 
  MODULE adaptive1d_module
  SOURCES ${sources}
  DEPENDENCIES ${dependencies})

target_link_libraries(adaptive1d_exe gfortran)

# add a dependency to the pseudo-target  
add_hpx_pseudo_dependencies(examples.adaptive1d adaptive1d_exe) 

