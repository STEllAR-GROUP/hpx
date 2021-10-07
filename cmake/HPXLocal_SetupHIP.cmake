# Copyright (c)      2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPXLocal_WITH_HIP AND NOT TARGET roc::hipblas)

  if(HPXLocal_WITH_CUDA)
    hpx_local_error(
      "Both HPXLocal_WITH_CUDA and HPXLocal_WITH_HIP are ON. Please choose one of \
    them for HPX to work properly"
    )
  endif(HPXLocal_WITH_CUDA)

  # Needed on rostam
  list(APPEND CMAKE_PREFIX_PATH $ENV{HIP_PATH}/lib/cmake/hip)
  list(APPEND CMAKE_PREFIX_PATH $ENV{DEVICE_LIB_PATH}/cmake/AMDDeviceLibs)
  list(APPEND CMAKE_PREFIX_PATH $ENV{DEVICE_LIB_PATH}/cmake/amd_comgr)
  list(APPEND CMAKE_PREFIX_PATH $ENV{DEVICE_LIB_PATH}/cmake/hsa-runtime64)
  # Setup hipblas (creates roc::hipblas)
  find_package(hipblas HINTS $ENV{HIPBLAS_ROOT} CONFIG)
  if(NOT hipblas_FOUND)
    hpx_local_warn(
      "Hipblas could not be found, the blas parts will therefore be disabled.\n\
      You can reconfigure specifying HIPBLAS_ROOT to enable hipblas"
    )
    set(HPXLocal_WITH_GPUBLAS OFF)
  else()
    set(HPXLocal_WITH_GPUBLAS ON)
    hpx_local_add_config_define(HPX_HAVE_GPUBLAS)
  endif()

  if(NOT HPXLocal_FIND_PACKAGE)
    # The cmake variables are supposed to be cached no need to redefine them
    hpx_local_add_config_define(HPX_HAVE_HIP)
  endif()

endif()
