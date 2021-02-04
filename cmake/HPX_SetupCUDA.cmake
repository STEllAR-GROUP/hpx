# Copyright (c) 2019 Ste||ar-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_CUDA AND NOT TARGET Cuda::cuda)

  find_package(CUDA REQUIRED)
  if(NOT HPX_FIND_PACKAGE)
    # The cmake variables are supposed to be cached no need to redefine them
    set(HPX_WITH_COMPUTE ON)
    hpx_add_config_define(HPX_HAVE_CUDA)
    hpx_add_config_define(HPX_HAVE_COMPUTE)
  endif()
  # keywords for target_link_libraries (cuda)
  set(CUDA_LINK_LIBRARIES_KEYWORD "PRIVATE")

  if(HPX_WITH_CUDA_CLANG AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    hpx_error(
      "To use Cuda Clang, please select Clang as your default C++ compiler"
    )
  endif()

  add_library(Cuda::cuda INTERFACE IMPORTED)
  target_include_directories(Cuda::cuda INTERFACE ${CUDA_INCLUDE_DIRS})

  if(NOT HPX_WITH_CUDA_CLANG)
    if(NOT MSVC)
      target_link_libraries(Cuda::cuda INTERFACE ${CUDA_LIBRARIES})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-w)
    else()
      set(CUDA_PROPAGATE_HOST_FLAGS OFF)
      target_link_directories(
        Cuda::cuda INTERFACE ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
      )
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
    target_link_directories(Cuda::cuda INTERFACE ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    target_link_libraries(Cuda::cuda INTERFACE cudart)
  endif()

  if(NOT HPX_FIND_PACKAGE)
    target_link_libraries(hpx_base_libraries INTERFACE Cuda::cuda)
  endif()
endif()
