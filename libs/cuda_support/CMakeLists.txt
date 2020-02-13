# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if (NOT TARGET Cuda::cuda)
    message(FATAL_ERROR "Cuda future support has been enabled, but the Cuda installation is not setup")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/cuda_support/include
set(cuda_support_headers
  hpx/cuda_support/cuda_future_helper.hpp  
)

# Default location is $HPX_ROOT/libs/cuda_support/include_compatibility
set(cuda_support_compat_headers)

set(cuda_support_sources
  cuda_target.cpp
  get_cuda_targets.cpp
)

include(HPX_AddModule)
add_hpx_module(cuda_support
  COMPATIBILITY_HEADERS OFF
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  SOURCES ${cuda_support_sources}
  HEADERS ${cuda_support_headers}
  COMPAT_HEADERS ${cuda_support_compat_headers}
  DEPENDENCIES
    hpx_affinity
    hpx_allocator_support
    hpx_assertion
    hpx_coroutines
    hpx_config
    hpx_errors
    hpx_execution
    hpx_functional
    hpx_memory
    hpx_performance_counters
    hpx_program_options
    hpx_runtime_configuration
    hpx_synchronization
    hpx_static_reinit
    hpx_type_support
    ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}
  CMAKE_SUBDIRS examples tests
)
