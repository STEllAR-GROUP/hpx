# Copyright (c) 2016 Thomas Heller
# Copyright (c) 2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs)

if(HPX_WITH_CUDA)
  set(example_programs ${example_programs}
      cublas_matmul
      data_copy
      hello_compute
      partitioned_vector
    )
  set(cublas_matmul_FLAGS DEPENDENCIES cublas)
#  set_source_files_properties(cublas_matmul.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
  set(data_copy_CUDA On)
  set(hello_compute_CUDA On)
  set(partitioned_vector_CUDA ON)
  set(partitioned_vector_FLAGS COMPONENT_DEPENDENCIES partitioned_vector)
endif()

foreach(example_program ${example_programs})
  if(${${example_program}_CUDA})
    set(sources
      ${example_program}.cu)
  else()
    set(sources
        ${example_program}.cpp)
  endif()

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/Compute/CUDA")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.compute.cuda.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.compute.cuda
                              examples.compute.cuda.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.compute.cuda.${example_program}
                              ${example_program}_exe)
endforeach()

