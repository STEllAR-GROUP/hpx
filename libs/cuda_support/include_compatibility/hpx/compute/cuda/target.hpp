//  Copyright (c) 2020 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/cuda_support/config/defines.hpp>
#include <hpx/cuda_support/target.hpp>

#if defined(HPX_CUDA_SUPPORT_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message(                                           \
    "The header hpx/compute/cuda/target.hpp is deprecated, \
    please include hpx/cuda_support/target.hpp instead")
#else
#warning                                                   \
    "The header hpx/compute/cuda/target.hpp is deprecated, \
    please include hpx/cuda_support/target.hpp instead"
#endif
#endif

namespace hpx { namespace compute { namespace cuda {
  using target = hpx::cuda::target;
}}}

