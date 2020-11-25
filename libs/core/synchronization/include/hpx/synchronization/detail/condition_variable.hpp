//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/cache_line_data.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution_base/agent_ref.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <boost/intrusive/slist.hpp>

#include <cstddef>
#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local { namespace detail {
    class condition_variable
    {
    public:
        HPX_NON_COPYABLE(condition_variable);

    private:
        using mutex_type = lcos::local::spinlock;

    private:
        // define data structures needed for intrusive slist container used for
        // the queues
        struct queue_entry
        {
            using hook_type = boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>>;

            queue_entry(hpx::execution_base::agent_ref ctx, void* q)
              : ctx_(ctx)
              , q_(q)
            {
            }

            hpx::execution_base::agent_ref ctx_;
            void* q_;
            hook_type slist_hook_;
        };

        using slist_option_type = boost::intrusive::member_hook<queue_entry,
            queue_entry::hook_type, &queue_entry::slist_hook_>;

        using queue_type = boost::intrusive::slist<queue_entry,
            slist_option_type, boost::intrusive::cache_last<true>,
            boost::intrusive::constant_time_size<true>>;

        struct reset_queue_entry
        {
            reset_queue_entry(queue_entry& e, queue_type& q)
              : e_(e)
              , last_(q.last())
            {
            }

            ~reset_queue_entry()
            {
                if (e_.ctx_)
                {
                    queue_type* q = static_cast<queue_type*>(e_.q_);
                    q->erase(last_);    // remove entry from queue
                }
            }

            queue_entry& e_;
            queue_type::const_iterator last_;
        };

    public:
        HPX_CORE_EXPORT condition_variable();

        HPX_CORE_EXPORT ~condition_variable();

        HPX_CORE_EXPORT bool empty(
            std::unique_lock<mutex_type> const& lock) const;

        HPX_CORE_EXPORT std::size_t size(
            std::unique_lock<mutex_type> const& lock) const;

        // Return false if no more threads are waiting (returns true if queue
        // is non-empty).
        HPX_CORE_EXPORT bool notify_one(std::unique_lock<mutex_type> lock,
            threads::thread_priority priority, error_code& ec = throws);

        HPX_CORE_EXPORT void notify_all(std::unique_lock<mutex_type> lock,
            threads::thread_priority priority, error_code& ec = throws);

        bool notify_one(
            std::unique_lock<mutex_type> lock, error_code& ec = throws)
        {
            return notify_one(
                std::move(lock), threads::thread_priority::default_, ec);
        }

        void notify_all(
            std::unique_lock<mutex_type> lock, error_code& ec = throws)
        {
            return notify_all(
                std::move(lock), threads::thread_priority::default_, ec);
        }

        HPX_CORE_EXPORT void abort_all(std::unique_lock<mutex_type> lock);

        HPX_CORE_EXPORT threads::thread_restart_state wait(
            std::unique_lock<mutex_type>& lock, char const* description,
            error_code& ec = throws);

        threads::thread_restart_state wait(
            std::unique_lock<mutex_type>& lock, error_code& ec = throws)
        {
            return wait(lock, "condition_variable::wait", ec);
        }

        HPX_CORE_EXPORT threads::thread_restart_state wait_until(
            std::unique_lock<mutex_type>& lock,
            hpx::chrono::steady_time_point const& abs_time,
            char const* description, error_code& ec = throws);

        threads::thread_restart_state wait_until(
            std::unique_lock<mutex_type>& lock,
            hpx::chrono::steady_time_point const& abs_time,
            error_code& ec = throws)
        {
            return wait_until(
                lock, abs_time, "condition_variable::wait_until", ec);
        }

        threads::thread_restart_state wait_for(
            std::unique_lock<mutex_type>& lock,
            hpx::chrono::steady_duration const& rel_time,
            char const* description, error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(), description, ec);
        }

        threads::thread_restart_state wait_for(
            std::unique_lock<mutex_type>& lock,
            hpx::chrono::steady_duration const& rel_time,
            error_code& ec = throws)
        {
            return wait_until(
                lock, rel_time.from_now(), "condition_variable::wait_for", ec);
        }

    private:
        template <typename Mutex>
        void abort_all(std::unique_lock<Mutex> lock);

        // re-add the remaining items to the original queue
        HPX_CORE_EXPORT void prepend_entries(
            std::unique_lock<mutex_type>& lock, queue_type& queue);

    private:
        queue_type queue_;
    };

    ///////////////////////////////////////////////////////////////////////////
    struct condition_variable_data;

    HPX_CORE_EXPORT void intrusive_ptr_add_ref(condition_variable_data* p);
    HPX_CORE_EXPORT void intrusive_ptr_release(condition_variable_data* p);

    struct condition_variable_data
    {
        typedef lcos::local::spinlock mutex_type;

        condition_variable_data()
          : count_(1)
        {
        }

        util::cache_aligned_data_derived<mutex_type> mtx_;
        util::cache_aligned_data_derived<detail::condition_variable> cond_;

    private:
        friend HPX_CORE_EXPORT void intrusive_ptr_add_ref(
            condition_variable_data*);
        friend HPX_CORE_EXPORT void intrusive_ptr_release(
            condition_variable_data*);

        hpx::util::atomic_count count_;
    };

}}}}    // namespace hpx::lcos::local::detail
