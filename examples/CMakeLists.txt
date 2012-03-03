# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    accumulator
    balancing
    bfs
    gtc
    performance_counters
    quickstart
    random_mem_access
   )

if(HDF5_CPP_FOUND AND HPX_CXX11_LAMBDAS AND HPX_CXX11_AUTO AND HDF5_THREAD_SAFE)
  set(subdirs ${subdirs}
      sheneos
     )
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(examples.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(examples examples.${subdir})
endforeach()

