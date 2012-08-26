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
    adaptive_dataflow
    allgather
    async_io
    balancing
    bfs
    bright_future
    cancelable_action
    dataflow
    distributed_test
    embarrassingly_parallel
    fortran_threadsafe
    gtc
    gtc_optimized
    gravity
    heartbeat
    hplpx
    interpolate1d
    interpolate3d
    jacobi
    nbody
    nqueen
    object_semaphore
    oclm_integration
    performance_counters
    queue
    quickstart
    qt
    random_mem_access
    sheneos
    spell_check
    startup_shutdown
    thread_aware_timer
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

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(examples.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(examples examples.${subdir})
endforeach()

