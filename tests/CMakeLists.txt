# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    performance
    correctness
    abi
   )

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests tests.${subdir})
endforeach()

