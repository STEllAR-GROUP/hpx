//  Copyright (c) 2020 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/compute/config/defines.hpp>
#include <hpx/include/compute.hpp>

#if HPX_COMPUTE_HAVE_DEPRECATION_WARNINGS
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/parallel/util/numa_binding_allocator.hpp is deprecated, \
    please include hpx/include/compute.hpp instead")
#else
#warning                                                                       \
    "The header hpx/parallel/util/numa_binding_allocator.hpp is deprecated, \
    please include hpx/include/compute.hpp instead"
#endif
#endif
