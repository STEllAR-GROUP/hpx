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
      balancing
      hplpx
      object_semaphore
      queue
      quickstart
     )

  if("${HPX_AGAS_VERSION}" STREQUAL "2")
    set(subdirs ${subdirs}
        dataflow
        heartbeat 
        throttle
       )
  endif()
else()
  set(subdirs
      accumulator
      queue
      quickstart
     )
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(examples.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(examples examples.${subdir})
endforeach()

