# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs
    point)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=8)
add_definitions(-DHPX_FUNCTION_ARGUMENT_LIMIT=11)

# for all targets specified above
foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()

set(sources
    gtc_client.cpp)

source_group("Source Files" FILES ${sources})

set(dependencies
    point_component
#    particle_component
    iostreams_component
    distributing_factory_component)

  # add example executable
add_hpx_executable(gtc_client
  MODULE gtc
  SOURCES ${sources}
  DEPENDENCIES ${dependencies}
  FOLDER "Examples/Applications/GTC")

# add a custom target for this example
add_hpx_pseudo_target(examples.gtc.point)
#add_hpx_pseudo_target(examples.gtc.particle)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.gtc
                            gtc_client_exe)

