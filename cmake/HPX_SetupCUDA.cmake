# Copyright (c) 2019 Ste||ar-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_CUDA AND NOT TARGET Cuda::cuda)

  if(HPX_WITH_CUDA_CLANG AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    hpx_error(
      "To use Cuda Clang, please select Clang as your default C++ compiler"
    )
  endif()

  # cuda_std_17 not recognized for previous versions
  cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

  set(HPX_WITH_GPUBLAS ON)
  hpx_add_config_define(HPX_HAVE_GPUBLAS)
  # Check CUDA standard
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    if (DEFINED CMAKE_CXX_STANDARD)
      set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    else()
      set(CMAKE_CUDA_STANDARD 17)
    endif()
  else()
    if(CMAKE_CUDA_STANDARD LESS 17)
      hpx_error(
        "You've set CMAKE_CUDA_STANDARD to ${CMAKE_CUDA_STANDARD}, which is less than 17 (the minimum required by HPX)"
      )
    endif()
  endif()

  enable_language(CUDA)

  if(NOT HPX_FIND_PACKAGE)
    # The cmake variables are supposed to be cached no need to redefine them
    set(HPX_WITH_COMPUTE ON)
    hpx_add_config_define(HPX_HAVE_CUDA)
    hpx_add_config_define(HPX_HAVE_COMPUTE)
  endif()

  # CUDA libraries used
  add_library(Cuda::cuda INTERFACE IMPORTED)
  # Toolkit targets like CUDA::cudart, CUDA::cublas, CUDA::cufft, etc. available
  find_package(CUDAToolkit MODULE REQUIRED)
  if (CUDAToolkit_FOUND)
    target_link_libraries(Cuda::cuda INTERFACE CUDA::cudart)
    target_link_libraries(Cuda::cuda INTERFACE CUDA::cublas)
  endif()
  # Flag not working for CLANG CUDA
  target_compile_features(Cuda::cuda INTERFACE $<$<CXX_COMPILER_ID:GNU>:
    cuda_std_${CMAKE_CUDA_STANDARD}
    >)
  set_target_properties(
    Cuda::cuda PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON
  )

  if(NOT HPX_WITH_CUDA_CLANG)
    if(NOT MSVC)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-w)
    else()
      # Windows
      set(CUDA_PROPAGATE_HOST_FLAGS OFF)
      set(CUDA_NVCC_FLAGS_DEBUG
          ${CUDA_NVCC_FLAGS_DEBUG};-D_DEBUG;-O0;-g;-G;-Xcompiler=-MDd;-Xcompiler=-Od;-Xcompiler=-Zi;-Xcompiler=-bigobj
      )
      set(CUDA_NVCC_FLAGS_RELWITHDEBINFO
          ${CUDA_NVCC_FLAGS_RELWITHDEBINFO};-DNDEBUG;-O2;-g;-Xcompiler=-MD,-O2,-Zi;-Xcompiler=-bigobj
      )
      set(CUDA_NVCC_FLAGS_MINSIZEREL
          ${CUDA_NVCC_FLAGS_MINSIZEREL};-DNDEBUG;-O1;-Xcompiler=-MD,-O1;-Xcompiler=-bigobj
      )
      set(CUDA_NVCC_FLAGS_RELEASE
          ${CUDA_NVCC_FLAGS_RELEASE};-DNDEBUG;-O2;-Xcompiler=-MD,-Ox;-Xcompiler=-bigobj
      )
    endif()
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-extended-lambda)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--default-stream per-thread)
  else()
    if(NOT HPX_FIND_PACKAGE)
      hpx_add_target_compile_option(-DBOOST_THREAD_USES_MOVE PUBLIC)
    endif()
  endif()

  if(NOT HPX_FIND_PACKAGE)
    target_link_libraries(hpx_base_libraries INTERFACE Cuda::cuda)
  endif()
endif()
