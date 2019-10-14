//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_FWD_HPP
#define HPX_RUNTIME_THREADS_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/coroutines/thread_enums.hpp>

namespace hpx
{
    namespace threads
    {
        namespace policies
        {
            struct scheduler_base;
            namespace detail {
                struct HPX_EXPORT affinity_data;
            }
        }

        namespace executors
        {
            struct HPX_EXPORT current_executor;
        }

        class HPX_EXPORT thread_pool_base;
        struct HPX_EXPORT thread_pool_init_parameters;
    }
}

#endif
