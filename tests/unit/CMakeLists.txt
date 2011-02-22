# Copyright (c) 2007-2008 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(type unit)

set(groups 
    sanity
    component_memory
    component_memory_block
    naming
    #lco
    #threadmanager
    performance_counters)

foreach(group ${groups})
  add_custom_target(tests.${type}.${group})
  add_subdirectory(${group})
  add_dependencies(tests.${type} tests.${type}.${group})
endforeach()

