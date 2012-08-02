# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2009-2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(HPX_RNPL)

# just recurse into all components subdirectories and execute the scripts there
set(subdirs
    dataflow
    stencil
    application)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=7)
add_definitions(-DHPX_FUNCTION_ARGUMENT_LIMIT=10)

# GCC 4.4.6 detects uninitialized variables in this application. These
# appear to be false positives as newer versions of GCC don't detect these.
if(NOT MSVC)
  if (040500 GREATER ${GCC_VERSION})
    hpx_remove_flag(-Werror=uninitialized)
  endif()
endif()

if(RNPL_FOUND)
  # FIXME: HPX_* identifiers please
  add_definitions(-DRNPL_FOUND)
  include_directories(${RNPL_INCLUDE_DIR})
endif()

foreach(subdir ${subdirs})
  add_subdirectory(${subdir})
endforeach()

# define build target for this directory
set(sources
    adaptive1d.cpp refine.cpp)

# define basic dependencies
set(dependencies
    adaptive1d_stencil_component
    adaptive1d_dataflow_component
    adaptive1d_application_lib)

source_group("Source Files" FILES ${sources})

add_hpx_executable(adaptive1d
  MODULE adaptive1d_module
  SOURCES ${sources}
  DEPENDENCIES ${dependencies}
  FOLDER "Examples/Applications/Adaptive1d")

# add a dependency to the pseudo-target
add_hpx_pseudo_dependencies(examples.adaptive1d adaptive1d_exe)

