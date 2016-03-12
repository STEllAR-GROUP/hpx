# Copyright (c) 2007-2016 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2013 Jeroen Habraken
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    component_storage
    containers
    iostreams
    performance_counters
    process
    security
   )

add_hpx_pseudo_target(components)

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(components.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(components components.${subdir})
endforeach()
