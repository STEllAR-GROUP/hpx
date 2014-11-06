#  Copyright (c) 2011 Thomas Heller
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    action_invoke_no_more_than
    copy_component
    get_gid
    get_ptr
    inheritance_2_classes_abstract
    inheritance_2_classes_concrete
    inheritance_3_classes_1_abstract
    inheritance_3_classes_2_abstract
    inheritance_3_classes_concrete
    migrate_component
    remote_object
    vector_algorithms
    vector_copy
    vector_handle_values
    vector
   )


set(action_invoke_no_more_than_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 4)
set(action_invoke_no_more_than_FLAGS
    DEPENDENCIES iostreams_component)

set(copy_component_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 2)

set(get_ptr_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 2)

set(migrate_component_PARAMETERS
    LOCALITIES 2
    THREADS_PER_LOCALITY 2)

set(remote_object_FLAGS
    DEPENDENCIES iostreams_component remote_object_component)
set(remote_object_PARAMETERS
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

set(vector_FLAGS DEPENDENCIES vector_component)
set(vector_algorithms_FLAGS DEPENDENCIES vector_component)

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
  add_hpx_pseudo_target(tests.unit.components.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.components
                              tests.unit.components.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.components.${test}
                              ${test}_test_exe)

endforeach()

