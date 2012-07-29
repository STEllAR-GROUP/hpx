# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    blocked_matrix_multiply
    householder_qr_factorization
    explicit_qr_algorithm
   )

#set(blocked_matrix_multiply_FLAGS DEPENDENCIES dataflow_component)

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources}) 
  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/HPXLA/${example_program}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.hpxla.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.hpxla
                              examples.hpxla.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.hpxla.${example_program}
                              ${example_program}_exe)
endforeach()

#set_property(TARGET blocked_matrix_multiply_exe APPEND
#  PROPERTY COMPILE_DEFINITIONS
#  "HPX_ACTION_ARGUMENT_LIMIT=10"
#  "HPX_COMPONENT_CREATE_ARGUMENT_LIMIT=10"
#  "HPX_FUNCTION_ARGUMENT_LIMIT=13")

