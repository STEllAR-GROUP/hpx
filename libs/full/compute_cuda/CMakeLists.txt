# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT (HPX_WITH_CUDA OR HPX_WITH_HIP)
   OR NOT HPX_WITH_ASYNC_CUDA
   OR NOT HPX_WITH_CUDA_COMPUTE
)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(compute_cuda_headers
    hpx/compute/cuda/allocator.hpp
    hpx/compute/cuda/concurrent_executor.hpp
    hpx/compute/cuda/concurrent_executor_parameters.hpp
    hpx/compute/cuda/default_executor.hpp
    hpx/compute/cuda/default_executor_parameters.hpp
    hpx/compute/cuda/detail/launch.hpp
    hpx/compute/cuda/detail/scoped_active_target.hpp
    hpx/compute/cuda.hpp
    hpx/compute/cuda/serialization/value_proxy.hpp
    hpx/compute/cuda/target_distribution_policy.hpp
    hpx/compute/cuda/target_ptr.hpp
    hpx/compute/cuda/traits/access_target.hpp
    hpx/compute/cuda/transfer.hpp
    hpx/compute/cuda/value_proxy.hpp
    hpx/compute/detail/get_proxy_type.hpp
    hpx/compute/detail/iterator.hpp
    hpx/compute/detail/target_distribution_policy.hpp
)

# cmake-format: off
set(compute_cuda_compat_headers)
# cmake-format: on

set(compute_cuda_sources)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(compute_cuda_optional_module_dependencies hpx_async_distributed)
endif()

include(HPX_AddModule)
add_hpx_module(
  full compute_cuda
  GLOBAL_HEADER_GEN OFF
  CUDA
  SOURCES ${compute_cuda_sources}
  HEADERS ${compute_cuda_headers}
  COMPAT_HEADERS ${compute_cuda_compat_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  MODULE_DEPENDENCIES hpx_compute hpx_async_cuda hpx_program_options
                      ${compute_cuda_optional_module_dependencies}
  CMAKE_SUBDIRS examples tests
)
