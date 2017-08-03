//  Copyright (c) 2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/threads/detail/scheduled_thread_pool.hpp>
#include <hpx/runtime/threads/detail/scheduled_thread_pool_impl.hpp>

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread pools of our choice
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::local_queue_scheduler<>>;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
#include <hpx/runtime/threads/policies/static_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::static_queue_scheduler<>>;
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/static_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::static_priority_queue_scheduler<>>;
#endif

#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::local_priority_queue_scheduler<hpx::compat::mutex,
        hpx::threads::policies::lockfree_fifo>>;
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::local_priority_queue_scheduler<hpx::compat::mutex,
        hpx::threads::policies::lockfree_lifo>>;

#if defined(HPX_HAVE_ABP_SCHEDULER)
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::local_priority_queue_scheduler<hpx::compat::mutex,
        hpx::threads::policies::lockfree_abp_fifo>>;
#endif

#if defined(HPX_HAVE_HIERARCHY_SCHEDULER)
#include <hpx/runtime/threads/policies/hierarchy_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::hierarchy_scheduler<>>;
#endif

#if defined(HPX_HAVE_PERIODIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/periodic_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::periodic_priority_queue_scheduler<>>;
#endif
