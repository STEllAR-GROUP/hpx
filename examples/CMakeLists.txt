# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    accumulator
    async_io
    balancing
    bfs
    cancelable_action
    gtc
    nbody
    performance_counters
    quickstart
   )

if(HDF5_CPP_FOUND)
  if(HDF5_THREAD_SAFE)
    set(subdirs ${subdirs}
        sheneos
       )
  endif()
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(examples.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(examples examples.${subdir})
endforeach()

