#  Copyright (c) 2014-2016 Hartmut Kaiser
#  Copyright (c) 2011 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(components)

set(tests)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests
      action_invoke_no_more_than
      copy_component
      get_gid
      get_ptr
      inheritance_2_classes_abstract
      inheritance_2_classes_concrete
      inheritance_3_classes_1_abstract
      inheritance_3_classes_2_abstract
      inheritance_2_classes_concrete_simple
      inheritance_3_classes_2_concrete
      inheritance_3_classes_concrete
      local_new
      migrate_component
      migrate_component_to_storage
      migrate_polymorphic_component
      new_
      new_binpacking
      new_colocated
  )

  set(action_invoke_no_more_than_PARAMETERS THREADS_PER_LOCALITY 4)
  set(action_invoke_no_more_than_FLAGS DEPENDENCIES iostreams_component)

  set(colocated_distribution_policy_PARAMETERS LOCALITIES 2
                                               THREADS_PER_LOCALITY 2
  )

  set(copy_component_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)

  set(get_ptr_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)

  set(migrate_component_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 2)
  set(migrate_component_FLAGS DEPENDENCIES iostreams_component)

  set(migrate_polymorphic_component_PARAMETERS LOCALITIES 2
                                               THREADS_PER_LOCALITY 2
  )
  set(migrate_polymorphic_component_FLAGS DEPENDENCIES iostreams_component)

  set(inheritance_2_classes_abstract_FLAGS DEPENDENCIES iostreams_component)

  set(inheritance_2_classes_concrete_FLAGS DEPENDENCIES iostreams_component)

  set(inheritance_3_classes_1_abstract_FLAGS DEPENDENCIES iostreams_component)

  set(inheritance_3_classes_2_abstract_FLAGS DEPENDENCIES iostreams_component)

  set(inheritance_3_classes_concrete_FLAGS DEPENDENCIES iostreams_component)

  set(inheritance_3_classes_2_concrete_FLAGS DEPENDENCIES iostreams_component)

  set(migrate_component_to_storage_FLAGS DEPENDENCIES unordered_component
                                         component_storage_component
  )

  set(new__PARAMETERS LOCALITIES 2)
  set(new_binpacking_PARAMETERS LOCALITIES 2)
  set(new_colocated_PARAMETERS LOCALITIES 2)
endif()

if(HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_TCP)
  set(tests ${tests} launch_process)

  # add executable needed for launch_process_test
  add_hpx_executable(
    launched_process_test INTERNAL_FLAGS
    SOURCES launched_process.cpp
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Components"
    COMPONENT_DEPENDENCIES launch_process_test_server
  )

  set(launch_process_FLAGS DEPENDENCIES iostreams_component process_component
                           launch_process_test_server_component
  )
  set(launch_process_PARAMETERS
      --launch=$<TARGET_FILE:launched_process_test>
      --hpx:expect-connecting-localities
      # Force use of the TCP parcelport
      --hpx:ini=hpx.parcel.tcp.priority=1000
      --hpx:ini=hpx.parcel.bootstrap=tcp
  )
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Components")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS} EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("component" ${test} ${${test}_PARAMETERS} RUN_SERIAL)

endforeach()

if(HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_TCP)
  add_hpx_pseudo_dependencies(
    tests.unit.component.launch_process launched_process_test
  )
endif()
