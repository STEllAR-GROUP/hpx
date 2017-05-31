# Copyright (c) 2017 Taeguk Kwon
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    local
   )

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.performance.parallel_algorithms.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.performance.parallel_algorithms
                              tests.performance.parallel_algorithms.${subdir})
endforeach()

