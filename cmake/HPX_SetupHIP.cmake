# Copyright (c)      2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_HIP AND NOT TARGET roc::hipblas)

  if(HPX_WITH_CUDA)
    hpx_error(
      "Both HPX_WITH_CUDA and HPX_WITH_HIP are ON. Please choose one of \
    them for HPX to work properly"
    )
  endif(HPX_WITH_CUDA)

  # Setup hipblas (creates roc::hipblas)
  find_package(hipblas HINTS $ENV{HIPBLAS_ROOT} CONFIG)
  if(NOT hipblas_FOUND)
    hpx_error(
      "Hipblas could not be found, please specify HIPBLAS_ROOT to point to the \
      correct location"
    )
  endif()
  target_include_directories(
    roc::hipblas SYSTEM INTERFACE ${hipblas_INCLUDE_DIRS}
  )

  set(HPX_WITH_COMPUTE ON)
  hpx_add_config_define(HPX_HAVE_COMPUTE)
  hpx_add_config_define(HPX_HAVE_HIP)

endif()
