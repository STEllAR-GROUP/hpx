# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    accumulator
    balancing
    quickstart
    object_semaphore
    random_mem_access
   )

if(${BOOST_MINOR_VERSION} GREATER 46)
  set(subdirs ${subdirs}
      bfs
      gtc
     )
endif()

if(HDF5_CPP_FOUND AND HPX_CXX11_LAMBDAS AND HPX_CXX11_AUTO)
  set(subdirs ${subdirs}
      sheneos
     )
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(examples.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(examples examples.${subdir})
endforeach()

