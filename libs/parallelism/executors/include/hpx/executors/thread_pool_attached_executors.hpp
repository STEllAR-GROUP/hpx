//  Copyright (c)      2020 Mikael Simberg
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/executors/restricted_thread_pool_executor.hpp>

namespace hpx { namespace parallel { namespace execution {
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
    using local_queue_attached_executor = restricted_thread_pool_executor;

    using static_queue_attached_executor = restricted_thread_pool_executor;

    using local_priority_queue_attached_executor =
        restricted_thread_pool_executor;

    using static_priority_queue_attached_executor =
        restricted_thread_pool_executor;
#endif
}}}    // namespace hpx::parallel::execution
