# Copyright (c) 2007-2008 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(group unit.boost)

set(subdirs 
    atomic
    cache
    lockfree
    move)

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(${group}.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(${group} ${group}.${subdir})
endforeach()

