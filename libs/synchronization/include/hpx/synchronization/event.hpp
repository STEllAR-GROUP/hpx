//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/synchronization/spinlock.hpp>

#include <atomic>
#include <mutex>
#include <utility>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local {
    /// Event semaphores can be used for synchronizing multiple threads that
    /// need to wait for an event to occur. When the event occurs, all threads
    /// waiting for the event are woken up.
    class event
    {
    private:
        typedef lcos::local::spinlock mutex_type;

    public:
        /// \brief Construct a new event semaphore
        event()
          : mtx_()
          , cond_()
          , event_(false)
        {
        }

        /// \brief Check if the event has occurred.
        bool occurred()
        {
            return event_.load(std::memory_order_acquire);
        }

        /// \brief Wait for the event to occur.
        void wait()
        {
            if (event_.load(std::memory_order_acquire))
                return;

            std::unique_lock<mutex_type> l(mtx_);
            wait_locked(l);
        }

        /// \brief Release all threads waiting on this semaphore.
        void set()
        {
            event_.store(true, std::memory_order_release);

            std::unique_lock<mutex_type> l(mtx_);
            set_locked(std::move(l));
        }

        /// \brief Reset the event
        void reset()
        {
            event_.store(false, std::memory_order_release);
        }

    private:
        void wait_locked(std::unique_lock<mutex_type>& l)
        {
            HPX_ASSERT(l.owns_lock());

            while (!event_.load(std::memory_order_acquire))
            {
                cond_.wait(l, "event::wait_locked");
            }
        }

        void set_locked(std::unique_lock<mutex_type> l)
        {
            HPX_ASSERT(l.owns_lock());

            // release the threads
            cond_.notify_all(std::move(l));
        }

        mutex_type mtx_;    ///< This mutex protects the queue.
        local::detail::condition_variable cond_;

        std::atomic<bool> event_;
    };
}}}    // namespace hpx::lcos::local
