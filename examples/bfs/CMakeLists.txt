# Copyright (c) 2011 Matt Anderson
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
# add a custom target for the standalone BFS example
add_hpx_pseudo_target(examples.bfs.standalone)

# the standalone subdirectory is special
add_subdirectory(standalone)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.bfs examples.bfs.standalone)

###############################################################################
# all other subdirectories follow the usual pattern

# just recurse into all components subdirectories and execute the scripts there
set(subdirs
    bfs_partitioned
    bfs_graph)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=8)

# for all targets specified above
foreach(subdir ${subdirs})
  add_subdirectory(${subdir})

  add_hpx_library_sources(${subdir}_exe
    GLOB GLOBS ${subdir}*.cpp)

  add_hpx_library_headers(${subdir}_exe
    GLOB GLOBS ${subdir}*.hpp )

  add_hpx_source_group(
    NAME ${subdir}_exe
    CLASS "Source Files"
    TARGETS ${${subdir}_exe_SOURCES})

  add_hpx_source_group(
    NAME ${subdir}_exe
    CLASS "Header Files"
    TARGETS ${${subdir}_exe_HEADERS})

  set(dependencies
    distributing_factory_component
    iostreams_component)

  # add example executable
  add_hpx_executable(${subdir}_client
    SOURCES ${${subdir}_exe_SOURCES}
    HEADERS ${${subdir}_exe_HEADERS}
    DEPENDENCIES ${subdir}_component ${dependencies}
    FOLDER "Examples/Bfs")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.bfs.${subdir})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.bfs examples.bfs.${subdir})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.bfs.${subdir} ${subdir}_client_exe)
endforeach()

