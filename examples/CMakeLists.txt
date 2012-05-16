# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    1d_hydro
    accumulator
    adaptive1d
    async_io
    balancing
    bfs
    bright_future
    cancelable_action
    dataflow
    distributed_test
    embarrassingly_parallel
    gtc
    heartbeat
    hplpx
    nqueen
    object_semaphore
    performance_counters
    queue
    quickstart
    random_mem_access
    startup_shutdown
    throttle
   )

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(${GCC_VERSION} GREATER 040500)
    set(subdirs ${subdirs}
        hpxla
       )
  endif()
endif()

if(${BOOST_MINOR_VERSION} GREATER 46)
  set(subdirs ${subdirs}
      geometry
     )
endif()

if(HDF5_CPP_FOUND)
  if(HDF5_FOUND)
    set(subdirs ${subdirs}
        interpolate1d
        interpolate3d
        gravity
       )
  endif()

  if(HDF5_THREAD_SAFE)
    set(subdirs ${subdirs}
        sheneos
       )

    if(CMAKE_Fortran_COMPILER AND HDF5_FORTRAN_FOUND)
      set(subdirs ${subdirs}
          neutron_star
         )
    endif()
  endif()
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(examples.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(examples examples.${subdir})
endforeach()

