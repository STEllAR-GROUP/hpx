# Copyright (c) 2007-2008 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(type unit)

set(groups 
    component_memory
    component_memory_block
    hardware
    lcos
    lockfree_deque
    naming
    performance_counters
    sanity
    serialization)
    #threadmanager

foreach(group ${groups})
  add_hpx_pseudo_target(tests.${type}.${group})
  add_subdirectory(${group})
  add_hpx_pseudo_dependencies(tests.${type} tests.${type}.${group})
endforeach()

