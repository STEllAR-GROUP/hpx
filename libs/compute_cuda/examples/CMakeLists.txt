# Copyright (c) 2016 Thomas Heller
# Copyright (c) 2016 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_CUDA_COMPUTE)
  return()
endif()

if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.modules.compute_cuda)
  add_hpx_pseudo_dependencies(examples.modules examples.modules.compute_cuda)
  if(HPX_WITH_TESTS
     AND HPX_WITH_TESTS_EXAMPLES
     AND HPX_COMPUTE_CUDA_WITH_TESTS
  )
    add_hpx_pseudo_target(tests.examples.modules.compute_cuda)
    add_hpx_pseudo_dependencies(
      tests.examples.modules tests.examples.modules.compute_cuda
    )
  endif()
else()
  return()
endif()

set(example_programs data_copy)

if(NOT HPX_WITH_CUDA_CLANG)
  # This example does not compile with the Cray clang compiler
  set(example_programs ${example_programs} hello_compute)
endif()

set(data_copy_PARAMETERS THREADS_PER_LOCALITY 4)
set(hello_compute_PARAMETERS THREADS_PER_LOCALITY 4)

# TODO: The partitioned_vector example works with neither with clang nor nvcc
# list(APPEND example_programs partitioned_vector)
# set(partitioned_vector_PARAMETERS THREADS_PER_LOCALITY 4)

include_directories(${CUDA_INCLUDE_DIRS})

set(data_copy_CUDA ON)
set(hello_compute_CUDA ON)
set(partitioned_vector_CUDA ON)
set(partitioned_vector_FLAGS DEPENDENCIES partitioned_vector_component)

foreach(example_program ${example_programs})
  if(${${example_program}_CUDA})
    set(sources ${example_program}.cu)
  else()
    set(sources ${example_program}.cpp)
  endif()

  if(${example_program}_CUDA_SOURCE)
    foreach(src ${${example_program}_CUDA_SOURCE})
      set(sources ${sources} ${src}.cu)
    endforeach()
  endif()

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Modules/Compute/CUDA"
  )

  add_hpx_example_target_dependencies("modules.compute_cuda" ${example_program})

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test(
      "modules.compute_cuda" ${example_program}
      ${${example_program}_PARAMETERS}
    )
  endif()
endforeach()
