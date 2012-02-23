# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs
    graph500)

# for all targets specified above
foreach(subdir ${subdirs})
  add_subdirectory(${subdir})

  set(sources
      graph500_util.cpp
      ${subdir}_client.cpp)

  source_group("Source Files" FILES ${sources})

  set(dependencies
      distributing_factory_component)

  # add example executable
  add_hpx_executable(${subdir}_client
    SOURCES ${sources}
    DEPENDENCIES ${subdir}_component ${dependencies}
    FOLDER "Examples/Bfs/${subdir}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.bfs.${subdir})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.bfs
                              examples.bfs.${subdir})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.bfs.${subdir}
                              ${subdir}_client_exe)
endforeach()

