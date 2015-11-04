# Copyright (c) 2011      Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs
    random_mem_access)

# for all targets specified above
foreach(subdir ${subdirs})
  add_subdirectory(${subdir})

  set(sources
      ${subdir}_client.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${subdir}_client
    MODULE random_mem_access
    SOURCES ${sources}
    DEPENDENCIES ${subdir}_component
    FOLDER "Examples/RandomMemoryAccess/random_mem_access")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.random_mem_access.${subdir}_dir)

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.random_mem_access
                              examples.random_mem_access.${subdir}_dir)

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.random_mem_access.${subdir}_dir
                              ${subdir}_client_exe)
endforeach()

