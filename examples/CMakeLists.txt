# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# NOTE: please do not comment out targets here and commit. If you do not want
# to build all the examples, you can explicitly specify which examples you
# would like to build on the command line, e.g. `make examples.had_amr3d'.
# It is important that all of these examples remain linked into the CMake
# dependency tree for testing purposes.

if(NOT MSVC)
  set(subdirs
      accumulator
      adaptive1d
      # Disabled by wash as this is randomly breaking things.
      # barnes_hut
      dataflow
      hplpx
      nqueen
      performance_counters
      queue
      quickstart
      random_mem_access
      heartbeat 
      throttle
      function
      math
      object_semaphore
      balancing)

  if(${BOOST_MINOR_VERSION} GREATER 46)
    set(subdirs ${subdirs} geometry
                           contact)
  endif() 

  if(HDF5_FOUND)
    set(subdirs ${subdirs} sheneos)
  endif()

  if(CMAKE_Fortran_COMPILER)
    set(subdirs ${subdirs} marduk)
  endif()

else()
  set(subdirs
      accumulator
      quickstart
      queue)

  if(${BOOST_MINOR_VERSION} GREATER 46)
    set(subdirs ${subdirs} geometry)
  endif() 
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(examples.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(examples examples.${subdir})
endforeach()

