# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT MSVC)
  set(subdirs
      accumulator
      adaptive1d
      balancing
      dataflow
      function
      heartbeat 
      hplpx
      math
      neutron_star
      nqueen
      object_semaphore
      performance_counters
      queue
      quickstart
      random_mem_access
      throttle
     )

  if(${BOOST_MINOR_VERSION} GREATER 46)
    set(subdirs ${subdirs} geometry
                           )
  endif() 

  if(HDF5_CPP_FOUND) 
    if(HDF5_FOUND) 
      set(subdirs ${subdirs} interpolate1d
                             interpolate3d)
    endif()
    if(HPX_CXX11_LAMBDAS)
      set(subdirs ${subdirs} sheneos)
    endif()
  endif()

  # wash: disabled as it breaks other components 
  #if(CMAKE_Fortran_COMPILER)
  #  set(subdirs ${subdirs} marduk)
  #endif()

else()
  set(subdirs
      accumulator
      quickstart
      queue
     )

  if(${BOOST_MINOR_VERSION} GREATER 46)
    set(subdirs ${subdirs} geometry)
  endif() 
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(examples.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(examples examples.${subdir})
endforeach()

