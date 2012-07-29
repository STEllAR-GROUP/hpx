# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs
    point_geometry)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=8)
add_definitions(-DHPX_FUNCTION_ARGUMENT_LIMIT=11)

# GCC 4.4.6 detects uninitialized variables in Boost.Geometry headers. These
# appear to be false positives as newer versions of GCC don't detect these.
if(NOT MSVC)
  if (040500 GREATER ${GCC_VERSION})
    hpx_remove_flag(-Werror=uninitialized)
  endif()
endif()

# for all targets specified above
foreach(subdir ${subdirs})
  add_subdirectory(${subdir})

  set(sources
      ${subdir}_client.cpp)

  source_group("Source Files" FILES ${sources})

  set(dependencies
      distributing_factory_component)

  # add example executable
  add_hpx_executable(${subdir}_client
    MODULE point_geometry
    SOURCES ${sources}
    DEPENDENCIES ${subdir}_component ${dependencies}
    FOLDER "Examples/Geometry/${subdir}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.geometry.${subdir})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.geometry
                              examples.geometry.${subdir})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.geometry.${subdir}
                              ${subdir}_client_exe)
endforeach()

