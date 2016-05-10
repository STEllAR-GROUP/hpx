//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_LOCAL_EVENT_HPP
#define HPX_LCOS_LOCAL_EVENT_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/assert.hpp>

#include <boost/atomic.hpp>

#include <mutex>

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
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
          : mtx_(), cond_(), event_(false)
        {}

        /// \brief Check if the event has occurred.
        bool occurred()
        {
            return event_.load(boost::memory_order_acquire);
        }

        /// \brief Wait for the event to occur.
        void wait()
        {
            if (event_.load(boost::memory_order_acquire))
                return;

            std::unique_lock<mutex_type> l(mtx_);
            wait_locked(l);
        }

        /// \brief Release all threads waiting on this semaphore.
        void set()
        {
            event_.store(true, boost::memory_order_release);

            std::unique_lock<mutex_type> l(mtx_);
            set_locked(std::move(l));
        }

        /// \brief Reset the event
        void reset()
        {
            event_.store(false, boost::memory_order_release);
        }

    private:
        void wait_locked(std::unique_lock<mutex_type>& l)
        {
            HPX_ASSERT(l.owns_lock());

            while (!event_.load(boost::memory_order_acquire))
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

        mutex_type mtx_;      ///< This mutex protects the queue.
        local::detail::condition_variable cond_;

        boost::atomic<bool> event_;
    };
}}}

#endif /*HPX_LCOS_LOCAL_EVENT_HPP*/
