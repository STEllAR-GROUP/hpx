//  Copyright (c) 2020 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_cuda/config/defines.hpp>
#include <hpx/async_cuda/target.hpp>

#if HPX_ASYNC_CUDA_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message("The header hpx/compute/cuda/target.hpp is deprecated, \
    please include hpx/async_cuda/target.hpp instead")
#else
#warning "The header hpx/compute/cuda/target.hpp is deprecated, \
    please include hpx/async_cuda/target.hpp instead"
#endif
#endif

namespace hpx { namespace compute { namespace cuda {
    using target = hpx::cuda::experimental::target;
}}}    // namespace hpx::compute::cuda
