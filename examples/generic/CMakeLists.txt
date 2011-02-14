# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there
set(hpx_GENERIC_SUBDIRS 
    generic)

foreach(hpx_GENERIC_SUBDIR ${hpx_GENERIC_SUBDIRS})
  add_subdirectory(${hpx_GENERIC_SUBDIR})
endforeach()

# define list of target names for this directory
set(hpx_GENERIC_EXAMPLES 
    generic) # example demonstrating generic_component

# for all targets specified above
foreach(hpx_GENERIC_EXAMPLE ${hpx_GENERIC_EXAMPLES})
  # define build target
  set(${hpx_GENERIC_EXAMPLE}_client_SOURCES 
      ${hpx_GENERIC_EXAMPLE}_client.cpp)

  # add example executable
  add_hpx_executable(${hpx_GENERIC_EXAMPLE}_client 
    SOURCES ${${hpx_GENERIC_EXAMPLE}_client_SOURCES}
    DEPENDENCIES ${hpx_GENERIC_EXAMPLE}_component)

  # add a custom target for this example 
  add_custom_target(examples.generic.${hpx_GENERIC_EXAMPLE})

  # add a dependency to the pseudo-target  
  add_dependencies(examples.generic.${hpx_GENERIC_EXAMPLE}
                   ${hpx_GENERIC_EXAMPLE}_client) 
endforeach()

