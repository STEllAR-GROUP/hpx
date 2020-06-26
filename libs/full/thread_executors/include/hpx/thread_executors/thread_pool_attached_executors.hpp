//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
#include <hpx/executors/thread_pool_attached_executors.hpp>

namespace hpx { namespace threads { namespace executors {
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
    using local_queue_attached_executor =
        parallel::execution::local_queue_attached_executor;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    using static_queue_attached_executor =
        parallel::execution::static_queue_attached_executor;
#endif

    using local_priority_queue_attached_executor =
        parallel::execution::local_priority_queue_attached_executor;

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    using static_priority_queue_attached_executor =
        parallel::execution::static_priority_queue_attached_executor;
#endif
}}}    // namespace hpx::threads::executors

#endif
