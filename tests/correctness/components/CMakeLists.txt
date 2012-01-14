#  Copyright (c) 2011 Thomas Heller
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    managed_component_lifetime
    remote_object_test
)

add_hpx_component(managed_test
    SOURCE_GLOB managed_test_component.c*
    HEADER_GLOB managed_test_component.h*
    DEPENDENCIES iostreams_component
    FOLDER "Tests/Correctness/Components"
    AUTOGLOB)

set(managed_component_lifetime_FLAGS
    DEPENDENCIES iostreams_component managed_test_component)

set(remote_object_test_FLAGS
    DEPENDENCIES iostreams_component remote_object_component)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Correctness/Components")

  # add a custom target for this example
  add_hpx_pseudo_target(tests.correctness.components.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.correctness.components
                              tests.correctness.components.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.correctness.components.${test}
                              ${test}_exe)
endforeach()
