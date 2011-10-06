# Copyright (c) 2011 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(components)

set(tests
    scoped_ref_to_local_object
    scoped_ref_to_remote_object
    local_embedded_ref_to_local_object
    local_embedded_ref_to_remote_object
    remote_embedded_ref_to_local_object
    remote_embedded_ref_to_remote_object
   )

set(scoped_ref_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component)

set(scoped_ref_to_remote_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component)

set(local_embedded_ref_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component)

set(local_embedded_ref_to_remote_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component)

set(remote_embedded_ref_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component)

set(remote_embedded_ref_to_remote_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}
                     MODULE agas
                     SOURCES ${sources}
                     ${${test}_FLAGS})

  # add a custom target for this example 
  add_hpx_pseudo_target(tests.correctness.agas.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.correctness.agas
                              tests.correctness.agas.${test})

  # add dependencies to pseudo-target 
  add_hpx_pseudo_dependencies(tests.correctness.agas.${test}
                              ${test}_exe)
endforeach() 
 
