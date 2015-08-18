#  Copyright (c) 2011 Thomas Heller
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    action_invoke_no_more_than
    copy_component
    distribution_policy_executor
    get_gid
    get_ptr
    inheritance_2_classes_abstract
    inheritance_2_classes_concrete
    inheritance_3_classes_1_abstract
    inheritance_3_classes_2_abstract
    inheritance_3_classes_concrete
    migrate_component
    migrate_component_to_storage
    new_
    new_binpacking
    new_colocated
    unordered_map
    vector_copy
    vector_for_each
    vector_handle_values
    vector_iter
    vector_move
    vector_transform_reduce
   )


set(action_invoke_no_more_than_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 4)
set(action_invoke_no_more_than_FLAGS
    DEPENDENCIES iostreams_component)

set(colocated_distribution_policy_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 2)

set(copy_component_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 2)

set(get_ptr_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 2)

set(migrate_component_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 2)

set(inheritance_2_classes_abstract_FLAGS
    DEPENDENCIES iostreams_component)

set(inheritance_2_classes_concrete_FLAGS
    DEPENDENCIES iostreams_component)

set(inheritance_3_classes_1_abstract_FLAGS
    DEPENDENCIES iostreams_component)

set(inheritance_3_classes_2_abstract_FLAGS
    DEPENDENCIES iostreams_component)

set(inheritance_3_classes_concrete_FLAGS
    DEPENDENCIES iostreams_component)

set(unordered_map_FLAGS
    DEPENDENCIES unordered_component)
set(unordered_map_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 2)

set(migrate_component_to_storage_FLAGS
    DEPENDENCIES unordered_component component_storage_component)

set(new__PARAMETERS LOCALITIES 2)
set(new_binpacking_PARAMETERS LOCALITIES 2)
set(new_colocated_PARAMETERS LOCALITIES 2)

set(vector_iter_FLAGS DEPENDENCIES vector_component)
set(vector_algorithms_FLAGS DEPENDENCIES vector_component)
set(vector_copy_FLAGS DEPENDENCIES vector_component)
set(vector_handle_values_FLAGS DEPENDENCIES vector_component)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Unit/Components")

  add_hpx_unit_test("components" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.component.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.component
                              tests.unit.component.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.component.${test}
                              ${test}_test_exe)

endforeach()

