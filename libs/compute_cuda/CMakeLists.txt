# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(HPX_WITH_CUDA)
  set(compute_cuda_headers
    hpx/compute/cuda/allocator.hpp
    hpx/compute/cuda/concurrent_executor.hpp
    hpx/compute/cuda/concurrent_executor_parameters.hpp
    hpx/compute/cuda/default_executor.hpp
    hpx/compute/cuda/default_executor_parameters.hpp
    hpx/compute/cuda/detail/launch.hpp
    hpx/compute/cuda/detail/scoped_active_target.hpp
    hpx/compute/cuda/get_targets.hpp
    hpx/compute/cuda.hpp
    hpx/compute/cuda/serialization/value_proxy.hpp
    hpx/compute/cuda/target_distribution_policy.hpp
    hpx/compute/cuda/target.hpp
    hpx/compute/cuda/target_ptr.hpp
    hpx/compute/cuda/traits/access_target.hpp
    hpx/compute/cuda/transfer.hpp
    hpx/compute/cuda/value_proxy.hpp
  )

  set(compute_cuda_compat_headers)

  set(compute_cuda_sources
    cuda_target.cpp
    get_cuda_targets.cpp
  )
endif()

include(HPX_AddModule)
add_hpx_module(compute_cuda
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  CUDA
  SOURCES ${compute_cuda_sources}
  HEADERS ${compute_cuda_headers}
  DEPENDENCIES
    hpx_allocator_support
    hpx_assertion
    hpx_compute
    hpx_config
    hpx_errors
    hpx_execution
    hpx_functional
    hpx_hardware
    hpx_plugin
    hpx_program_options
    hpx_topology
    hpx_type_support
    hpx_serialization
  CMAKE_SUBDIRS examples tests
)
