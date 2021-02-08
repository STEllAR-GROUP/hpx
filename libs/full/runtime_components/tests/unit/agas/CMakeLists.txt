# Copyright (c) 2011 Bryce Adelstein-Lelbach
# Copyright (c) 2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# add subdirectories
add_subdirectory(components)

set(tests
    find_clients_from_prefix
    find_ids_from_prefix
    get_colocation_id
    local_address_rebind
    local_embedded_ref_to_local_object
    refcnted_symbol_to_local_object
    scoped_ref_to_local_object
    split_credit
    uncounted_symbol_to_local_object
)

set(find_ids_from_prefix_PARAMETERS LOCALITIES 2)
set(find_clients_from_prefix_PARAMETERS LOCALITIES 2)

set(get_colocation_id_PARAMETERS LOCALITIES 2)

set(local_address_rebind_FLAGS DEPENDENCIES iostreams_component
                               simple_mobile_object_component
)
set(local_address_rebind_PARAMETERS THREADS_PER_LOCALITY 4)

set(scoped_ref_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
    managed_refcnt_checker_component
)
set(scoped_ref_to_local_object_PARAMETERS THREADS_PER_LOCALITY 4)

set(local_embedded_ref_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
    managed_refcnt_checker_component
)
set(local_embedded_ref_to_local_object_PARAMETERS THREADS_PER_LOCALITY 4)

set(refcnted_symbol_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
    managed_refcnt_checker_component
)
set(refcnted_symbol_to_local_object_PARAMETERS THREADS_PER_LOCALITY 4)

set(uncounted_symbol_to_local_object_FLAGS
    DEPENDENCIES simple_refcnt_checker_component
    managed_refcnt_checker_component
)
set(uncounted_symbol_to_local_object_PARAMETERS THREADS_PER_LOCALITY 4)

set(split_credit_FLAGS DEPENDENCIES simple_refcnt_checker_component
                       managed_refcnt_checker_component
)
set(split_credit_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)

if(HPX_WITH_NETWORKING)
  set(tests
      ${tests}
      credit_exhaustion
      local_embedded_ref_to_remote_object
      remote_embedded_ref_to_local_object
      remote_embedded_ref_to_remote_object
      refcnted_symbol_to_remote_object
      uncounted_symbol_to_remote_object
      scoped_ref_to_remote_object
  )
  set(credit_exhaustion_FLAGS DEPENDENCIES simple_refcnt_checker_component
                              managed_refcnt_checker_component
  )
  set(credit_exhaustion_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)

  set(local_embedded_ref_to_remote_object_FLAGS
      DEPENDENCIES simple_refcnt_checker_component
      managed_refcnt_checker_component
  )
  set(local_embedded_ref_to_remote_object_PARAMETERS LOCALITIES 2
                                                     THREADS_PER_LOCALITY 2
  )

  set(remote_embedded_ref_to_local_object_FLAGS
      DEPENDENCIES simple_refcnt_checker_component
      managed_refcnt_checker_component
  )
  set(remote_embedded_ref_to_local_object_PARAMETERS LOCALITIES 2
                                                     THREADS_PER_LOCALITY 2
  )

  set(remote_embedded_ref_to_remote_object_FLAGS
      DEPENDENCIES simple_refcnt_checker_component
      managed_refcnt_checker_component
  )
  set(remote_embedded_ref_to_remote_object_PARAMETERS LOCALITIES 2
                                                      THREADS_PER_LOCALITY 2
  )

  set(scoped_ref_to_remote_object_FLAGS
      DEPENDENCIES simple_refcnt_checker_component
      managed_refcnt_checker_component
  )
  set(scoped_ref_to_remote_object_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY
                                             2
  )

  set(refcnted_symbol_to_remote_object_FLAGS
      DEPENDENCIES simple_refcnt_checker_component
      managed_refcnt_checker_component
  )
  set(refcnted_symbol_to_remote_object_PARAMETERS LOCALITIES 2
                                                  THREADS_PER_LOCALITY 2
  )

  set(uncounted_symbol_to_remote_object_FLAGS
      DEPENDENCIES simple_refcnt_checker_component
      managed_refcnt_checker_component
  )
  set(uncounted_symbol_to_remote_object_PARAMETERS LOCALITIES 2
                                                   THREADS_PER_LOCALITY 2
  )
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Full/AGAS")

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("modules.runtime_components" ${test} ${${test}_PARAMETERS})
endforeach()
