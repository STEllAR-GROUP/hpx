# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    overhead
   )

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.performance.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.performance tests.performance.${subdir})
endforeach()

