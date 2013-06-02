# Copyright (c) 2007-2010 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2013 Jeroen Habraken
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    binpacking_factory
    dataflow
    distributing_factory
    iostreams
    memory
    remote_object
   )

if(PAPI_FOUND)
  set(subdirs ${subdirs} papi_counters)
endif()

if(NOT HPX_HAVE_SECURITY AND SODIUM_FOUND)
  set(subdirs ${subdirs} security)
endif()

add_hpx_pseudo_target(components)

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(components.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(components components.${subdir})
endforeach()

