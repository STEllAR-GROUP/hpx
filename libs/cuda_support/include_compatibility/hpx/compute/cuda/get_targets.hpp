//  Copyright (c) 2020 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/cuda_support/config/defines.hpp>
#include <hpx/cuda_support/get_targets.hpp>

#if defined(HPX_CUDA_SUPPORT_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message("The header hpx/compute/cuda/get_targets.hpp is deprecated, \
    please include hpx/cuda_support/get_targets.hpp instead")
#else
#warning "The header hpx/compute/cuda/get_targets.hpp is deprecated, \
    please include hpx/cuda_support/get_targets.hpp instead"
#endif
#endif
