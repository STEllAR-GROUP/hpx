# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(hpx_ACCUMULATOR_SUBDIRS 
    accumulator
    simple_accumulator)

foreach(hpx_ACCUMULATOR_SUBDIR ${hpx_ACCUMULATOR_SUBDIRS})
  add_subdirectory(${hpx_ACCUMULATOR_SUBDIR})
endforeach()

# define list of target names for this directory
set(hpx_ACCUMULATOR_EXAMPLES 
    accumulator          # accumulator_client example
    simple_accumulator)  # simple_accumulator_client example

# for all targets specified above
foreach(hpx_ACCUMULATOR_EXAMPLE ${hpx_ACCUMULATOR_EXAMPLES})
  # define build target
  set(${hpx_ACCUMULATOR_EXAMPLE}_client_SOURCES 
      ${hpx_ACCUMULATOR_EXAMPLE}_client.cpp)

  # add example executable
  add_hpx_executable(${hpx_ACCUMULATOR_EXAMPLE}_client 
    SOURCES ${${hpx_ACCUMULATOR_EXAMPLE}_client_SOURCES}
    DEPENDENCIES ${hpx_ACCUMULATOR_EXAMPLE}_component)

  # add a custom target for this example 
  add_custom_target(examples.accumulator.${hpx_ACCUMULATOR_EXAMPLE})

  # add a dependency to the pseudo-target  
  add_dependencies(examples.accumulator.${hpx_ACCUMULATOR_EXAMPLE}
                   ${hpx_ACCUMULATOR_EXAMPLE}_client_exe) 
endforeach()

