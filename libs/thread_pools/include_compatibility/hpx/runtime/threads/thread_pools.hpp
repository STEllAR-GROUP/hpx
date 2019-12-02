//  Copyright (c) 2019 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/thread_pools/config/defines.hpp>
#include <hpx/thread_data/thread_pool_base.hpp>
#include <hpx/thread_pools/scheduled_thread_pool.hpp>

#if defined(HPX_THREAD_POOLS_HAVE_DEPRECATION_WARNINGS)
#if defined(HPX_MSVC)
#pragma message(                                                               \
    "The header hpx/runtime/threads/thread_pools.hpp is deprecated, \
    please include hpx/thread_pools/scheduled_thread_pool.hpp \
    or hpx/thread_data/thread_pool_base.hpp instead")
#else
#warning "The header hpx/runtime/threads/thread_pools.hpp is deprecated, \
    please include hpx/thread_pools/scheduled_thread_pool.hpp \
    or hpx/thread_data/thread_pool_base.hpp instead"
#endif
#endif
