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
    refcnted_symbol_to_local_object
    refcnted_symbol_to_remote_object
    uncounted_symbol_to_local_object
    uncounted_symbol_to_remote_object
    gid_type
    credit_exhaustion
    split_credit
   )

set(scoped_ref_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

set(scoped_ref_to_remote_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

set(local_embedded_ref_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

set(local_embedded_ref_to_remote_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

set(remote_embedded_ref_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

set(remote_embedded_ref_to_remote_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

set(refcnted_symbol_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

set(refcnted_symbol_to_remote_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

set(uncounted_symbol_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

set(uncounted_symbol_to_remote_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

set(credit_exhaustion_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

set(split_credit_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
                 managed_refcnt_checker_component)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Unit/AGAS")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.agas.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.agas
                              tests.unit.agas.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.agas.${test}
                              ${test}_test_exe)
endforeach()

