# Copyright (c) 2014-2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add subdirectories
set(subdirs
    executors
   )

if(HPX_WITH_DATAPAR)
  set(subdirs ${subdirs}
      datapar
     )
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.regressions.parallel.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(
    tests.regressions.parallel tests.regressions.parallel.${subdir})
endforeach()

