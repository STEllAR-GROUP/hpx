# Copyright (c) 2007-2015 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add subdirectories
set(subdirs
    block
    datapar_algorithms
    executors
    segmented_algorithms
   )

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.unit.parallel.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.unit.parallel tests.unit.parallel.${subdir})
endforeach()
