# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs
    bfs bfs_tm)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=8)

# for all targets specified above
foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()

set(sources
    bfs_client.cpp
    validate.cpp)

source_group("Source Files" FILES ${sources})

set(dependencies
    distributing_factory_component
    bfs_component
    bfs_tm_component
    iostreams_component)

# add example executable
add_hpx_executable(bfs_client
  MODULE bfs
  SOURCES ${sources}
  DEPENDENCIES ${dependencies})

  # add a custom target for this example
  add_hpx_pseudo_target(examples.bfs.${subdir})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.bfs
                              examples.bfs.${subdir})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.bfs.${subdir}
                              bfs_client_exe)

