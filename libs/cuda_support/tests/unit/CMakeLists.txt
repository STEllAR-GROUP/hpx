# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if (NOT HPX_WITH_CUDA_SUPPORT)
  return()
endif()

set(tests
    cublas_matmul
    cuda_future
)

set(cublas_matmul_PARAMETERS THREADS_PER_LOCALITY 4)
set(cuda_future_PARAMETERS   THREADS_PER_LOCALITY 4)


set(cublas_matmul_FLAGS
    DEPENDENCIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

set(cuda_future_CUDA_SOURCE saxpy trivial_demo)

foreach(test ${tests})
  set(sources
    ${test}.cpp)

  if(${test}_CUDA_SOURCE)
    foreach(src ${${test}_CUDA_SOURCE})
      set(sources
        ${sources} ${src}.cu)
    endforeach()
  endif()

  source_group("Source Files" FILES ${sources})

  # add test executable
  add_hpx_executable(${test}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    ${${test}_FLAGS}
    DEPENDENCIES hpx_cuda_support # hpx::cuda 
    ${${test_program}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/cuda_support")

  add_hpx_unit_test("modules.cuda_support" ${test} ${${test}_PARAMETERS})

endforeach()
