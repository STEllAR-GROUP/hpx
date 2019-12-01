//  Copyright (c) 2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/threads/detail/scheduled_thread_pool.hpp>
#include <hpx/runtime/threads/detail/scheduled_thread_pool_impl.hpp>

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread pools of our choice
#if defined(HPX_HAVE_LOCAL_SCHEDULER)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::policies::local_queue_scheduler<>;
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::local_queue_scheduler<>>;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
#include <hpx/runtime/threads/policies/static_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::policies::static_queue_scheduler<>;
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::static_queue_scheduler<>>;
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/static_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::policies::static_priority_queue_scheduler<>;
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::static_priority_queue_scheduler<>>;
#endif

#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::policies::local_priority_queue_scheduler<
    std::mutex, hpx::threads::policies::lockfree_fifo>;
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::local_priority_queue_scheduler<std::mutex,
        hpx::threads::policies::lockfree_fifo>>;
#if defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
template class HPX_EXPORT hpx::threads::policies::local_priority_queue_scheduler<
    std::mutex, hpx::threads::policies::lockfree_lifo>;
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::local_priority_queue_scheduler<std::mutex,
        hpx::threads::policies::lockfree_lifo>>;
#endif

#if defined(HPX_HAVE_LOCAL_WORKSTEALING_SCHEDULER)
#include <hpx/runtime/threads/policies/local_workstealing_scheduler.hpp>
template class HPX_EXPORT hpx::threads::policies::local_workstealing_scheduler<>;
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::local_workstealing_scheduler<>>;
template class HPX_EXPORT hpx::threads::policies::local_workstealing_scheduler<
    std::mutex, hpx::threads::policies::lockfree_lifo>;
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::local_workstealing_scheduler<std::mutex,
        hpx::threads::policies::lockfree_lifo>>;
#endif

#if defined(HPX_HAVE_ABP_SCHEDULER) && defined(HPX_HAVE_CXX11_STD_ATOMIC_128BIT)
template class HPX_EXPORT hpx::threads::policies::local_priority_queue_scheduler<
    std::mutex, hpx::threads::policies::lockfree_abp_fifo>;
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::local_priority_queue_scheduler<std::mutex,
        hpx::threads::policies::lockfree_abp_fifo>>;
template class HPX_EXPORT hpx::threads::policies::local_priority_queue_scheduler<
    std::mutex, hpx::threads::policies::lockfree_abp_lifo>;
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::local_priority_queue_scheduler<std::mutex,
        hpx::threads::policies::lockfree_abp_lifo>>;
#endif

#if defined(HPX_HAVE_SHARED_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/shared_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::policies::shared_priority_queue_scheduler<>;
template class HPX_EXPORT hpx::threads::detail::scheduled_thread_pool<
    hpx::threads::policies::shared_priority_queue_scheduler<>>;
#endif
