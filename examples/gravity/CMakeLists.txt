# Copyright (c) 2007-2010 Hartmut Kaiser
# Copyright (c) 2009-2010 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Adrian Serio
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=7)

# define build target for this directory
set(sources
    gravity_hpx.cpp)

# define basic dependencies
#set(dependencies )

source_group("Source Files" FILES ${sources})

add_hpx_executable(gravity_hpx
  MODULE gravity_module
  SOURCES ${sources}
  DEPENDENCIES ${dependencies})

# add a dependency to the pseudo-target
add_hpx_pseudo_target(examples.gravity_hpx)
add_hpx_pseudo_dependencies(examples.gravity_hpx gravity_hpx_exe)

