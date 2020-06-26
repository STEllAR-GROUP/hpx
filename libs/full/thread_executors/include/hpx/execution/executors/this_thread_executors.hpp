//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
#include <hpx/thread_executors/this_thread_executors.hpp>
#include <hpx/thread_executors/thread_execution.hpp>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_STATIC_SCHEDULER)
    using this_thread_static_queue_executor =
        threads::executors::this_thread_static_queue_executor;
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    using this_thread_static_priority_queue_executor =
        threads::executors::this_thread_static_priority_queue_executor;
#endif
}}}    // namespace hpx::parallel::execution

#endif
