# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs 
    interpolate1d)

# for all targets specified above
foreach(subdir ${subdirs})
  add_subdirectory(${subdir})

  set(sources
      ${subdir}_client.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${subdir}_client 
    MODULE interpolate1d
    SOURCES ${sources} 
    DEPENDENCIES ${subdir}_component)

  # add a custom target for this example 
  add_hpx_pseudo_target(examples.interpolate1d.${subdir})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.interpolate1d
                              examples.interpolate1d.${subdir})

  # add dependencies to pseudo-target 
  add_hpx_pseudo_dependencies(examples.interpolate1d.${subdir}
                              ${subdir}_client_exe) 
endforeach()

if(HDF5_FOUND)
  add_hpx_executable(create_testdata
    MODULE interpolate1d
    SOURCES create_testdata.cpp
    DEPENDENCIES ${HDF5_LIBRARY}
    NOLIBS)

  # add a custom target for this example 
  add_hpx_pseudo_target(examples.interpolate1d.create_testdata)

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.interpolate1d
                              examples.interpolate1d.create_testdata)

  # add dependencies to pseudo-target 
  add_hpx_pseudo_dependencies(examples.interpolate1d.create_testdata
                              create_testdata_exe)
endif() 

