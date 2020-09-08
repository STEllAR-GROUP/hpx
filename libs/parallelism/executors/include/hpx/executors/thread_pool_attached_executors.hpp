//  Copyright (c)      2020 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/executors/restricted_thread_pool_executor.hpp>

namespace hpx { namespace parallel { namespace execution {
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    using local_queue_attached_executor = restricted_thread_pool_executor;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    using static_queue_attached_executor = restricted_thread_pool_executor;
#endif

    using local_priority_queue_attached_executor =
        restricted_thread_pool_executor;

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    using static_priority_queue_attached_executor =
        restricted_thread_pool_executor;
#endif
#endif
}}}    // namespace hpx::parallel::execution
