# Copyright (c) 2020 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(benchmarks cuda_executor_throughput)

set(cuda_executor_throughput_PARAMETERS THREADS_PER_LOCALITY 1)
set(cuda_executor_throughput_FLAGS DEPENDENCIES Cuda::cuda
                                   ${CUDA_CUBLAS_LIBRARIES}
)

foreach(benchmark ${benchmarks})

  set(sources ${benchmark}.cpp)

  source_group("Source Files" FILES ${sources})

  # add benchmark executable
  add_hpx_executable(
    ${benchmark}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL ${${benchmark}_FLAGS}
    DEPENDENCIES hpx_async_cuda ${${benchmark}_FLAGS}
    FOLDER "Benchmarks/Modules/Full/AsyncCuda"
  )

  # add a custom target for this benchmark
  add_hpx_performance_test(
    "modules.async_cuda" ${benchmark} ${${benchmark}_PARAMETERS}
  )

endforeach()
