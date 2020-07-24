# Copyright (c) 2014 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

set(example_programs
    1d_stencil_1
    1d_stencil_2
    1d_stencil_3
    1d_stencil_4
    1d_stencil_4_parallel
    1d_stencil_5
    1d_stencil_6
    1d_stencil_7
    1d_stencil_8
    1d_stencil_channel
)

if(HPX_WITH_APEX)
  set(example_programs ${example_programs} 1d_stencil_4_repart
                       1d_stencil_4_throttle
  )

  set(1d_stencil_4_repart_PARAMETERS THREADS_PER_LOCALITY 4)
  set(1d_stencil_4_throttle_PARAMETERS THREADS_PER_LOCALITY 4)
endif()

if(HPX_WITH_EXAMPLES_OPENMP)
  set(example_programs ${example_programs} 1d_stencil_1_omp 1d_stencil_3_omp)

  set(disabled_tests 1d_stencil_1_omp 1d_stencil_3_omp)
endif()

set(1d_stencil_1_PARAMETERS THREADS_PER_LOCALITY 4)
set(1d_stencil_2_PARAMETERS THREADS_PER_LOCALITY 4)
set(1d_stencil_3_PARAMETERS THREADS_PER_LOCALITY 4)
set(1d_stencil_4_PARAMETERS THREADS_PER_LOCALITY 4)
set(1d_stencil_4_parallel_PARAMETERS THREADS_PER_LOCALITY 4)
set(1d_stencil_5_PARAMETERS THREADS_PER_LOCALITY 4)
set(1d_stencil_6_PARAMETERS THREADS_PER_LOCALITY 4)
set(1d_stencil_7_PARAMETERS THREADS_PER_LOCALITY 4)
set(1d_stencil_8_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(example_program ${example_programs})

  set(${example_program}_FLAGS COMPONENT_DEPENDENCIES iostreams)

  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/1D Stencil/${example_program}"
  )

  add_hpx_example_target_dependencies("1d_stencil" ${example_program})

  if(HPX_WITH_TESTS
     AND HPX_WITH_TESTS_EXAMPLES
     AND NOT ("${example_program}" IN_LIST disabled_tests)
  )

    add_hpx_example_test(
      "1d_stencil" ${example_program} ${${example_program}_PARAMETERS}
    )

  endif()

endforeach()

if(HPX_WITH_EXAMPLES_OPENMP)
  set_target_properties(
    1d_stencil_1_omp PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
  )
  set_target_properties(
    1d_stencil_3_omp PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
  )
  if(NOT MSVC)
    set_target_properties(
      1d_stencil_1_omp PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}"
    )
    set_target_properties(
      1d_stencil_3_omp PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}"
    )
  endif()
endif()
