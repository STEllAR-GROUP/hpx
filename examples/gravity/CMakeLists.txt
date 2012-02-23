# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2009-2010 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Adrian Serio
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=10)
add_definitions(-DHPX_FUNCTION_LIMIT=13)

include_directories (${HDF5_INCLUDE_DIR} ${HDF5_CPP_INCLUDE_DIR})

# define build target for this directory
set(sources
    gravity_hpx.cpp
    HDF5_reader.cpp
    HDF5_writer.cpp)

set(headers
    gravity.hpp)

# define basic dependencies
set(dependencies 
    ${HDF5_CPP_LIBRARY} ${HDF5_LIBRARY})

source_group("Source Files" FILES ${sources})

add_hpx_executable(gravity_hpx
  MODULE gravity_module
  SOURCES ${sources}
  HEADERS ${headers}
  DEPENDENCIES ${dependencies}
  FOLDER "Examples/Gravity")

# add a dependency to the pseudo-target
add_hpx_pseudo_dependencies(examples.gravity gravity_hpx_exe)

