# Copyright (c) 2007-2010 Hartmut Kaiser
# Copyright (c) 2009-2010 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs 
    mesh
    amr_c
    amr_c_test)

if(RNPL_FOUND)
  # FIXME: HPX_* identifiers please
  add_definitions(-DRNPL_FOUND)
  include_directories(${RNPL_INCLUDE_DIR})
endif()

foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()

# define build target for this directory
set(sources
    marduk.cpp level.f util.f)

# define basic dependencies
set(dependencies
    ssmp_amr3d_c_component)

source_group("Source Files" FILES ${sources})

add_hpx_executable(marduk 
  MODULE ssmp_amr3d
  SOURCES ${sources}
  DEPENDENCIES ${dependencies})

# add a dependency to the pseudo-target  
add_hpx_pseudo_dependencies(examples.marduk marduk_exe) 

