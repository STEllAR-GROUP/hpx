# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs
    accumulator
    simple_accumulator)

# for all targets specified above
foreach(subdir ${subdirs})
  add_subdirectory(${subdir})

  set(sources
      ${subdir}_client.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${subdir}_client
    SOURCES ${sources}
    DEPENDENCIES ${subdir}_component
    FOLDER "Examples/Quickstart/${subdir}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.accumulator.${subdir})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.accumulator
                              examples.accumulator.${subdir})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.accumulator.${subdir}
                              ${subdir}_client_exe)
endforeach()

