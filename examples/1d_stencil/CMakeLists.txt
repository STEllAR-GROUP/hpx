# Copyright (c) 2014 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    1d_stencil_1
    1d_stencil_2
    1d_stencil_3
    1d_stencil_4
    1d_stencil_4_parallel
    1d_stencil_5
    1d_stencil_6
    1d_stencil_7
    1d_stencil_8)

if(HPX_WITH_APEX)
  set(example_programs ${example_programs}
      1d_stencil_4_repart
      1d_stencil_4_throttle
     )
endif()

if(HPX_WITH_OPENMP)
  set(example_programs ${example_programs}
      1d_stencil_1_omp
      1d_stencil_3_omp)
endif()

foreach(example_program ${example_programs})

  set(${example_program}_FLAGS COMPONENT_DEPENDENCIES iostreams)

  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/1D Stencil/${example_program}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.1d_stencil.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.1d_stencil
                              examples.1d_stencil.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.1d_stencil.${example_program}
                              ${example_program}_exe)
endforeach()

if(HPX_WITH_OPENMP)
  set_target_properties(1d_stencil_1_omp_exe
    PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
  set_target_properties(1d_stencil_3_omp_exe
    PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
  if(NOT MSVC)
    set_target_properties(1d_stencil_1_omp_exe
      PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    set_target_properties(1d_stencil_3_omp_exe
      PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()
endif()
