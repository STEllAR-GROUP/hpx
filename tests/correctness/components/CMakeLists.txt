
set(tests
    managed_component_lifetime
)

add_hpx_component(
    managed_test
    MODULE tests_correctness_component
    SOURCES managed_test_component.cpp
    HEADERS managed_test_component.hpp
    DEPENDENCIES iostreams_component
)


set(managed_component_lifetime_FLAGS
    DEPENDENCIES iostreams_component managed_test_component
)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}
                     SOURCES ${sources}
                     ${${test}_FLAGS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.correctness.components.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.correctness.components
                              tests.correctness.components.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.correctness.components.${test}
                              ${test}_exe)
endforeach()
