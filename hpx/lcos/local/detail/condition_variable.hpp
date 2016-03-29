//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_LOCAL_DETAIL_CONDITION_VARIABLE_HPP
#define HPX_LCOS_LOCAL_DETAIL_CONDITION_VARIABLE_HPP

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/util/date_time_chrono.hpp>

#include <boost/intrusive/slist.hpp>
#include <boost/thread/locks.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local { namespace detail
{
    class condition_variable
    {
        HPX_NON_COPYABLE(condition_variable);

    private:
        typedef lcos::local::spinlock mutex_type;

        class relock_guard
        {
            HPX_NON_COPYABLE(relock_guard)

        public:
            explicit relock_guard(boost::unique_lock<mutex_type>& l)
              : l_(l)
            {}

            ~relock_guard()
            {
                l_.lock();
            }

        private:
            boost::unique_lock<mutex_type>& l_;
        };

    private:
        // define data structures needed for intrusive slist container used for
        // the queues
        struct queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            queue_entry(threads::thread_id_repr_type const& id, void* q)
              : id_(id), q_(q)
            {}

            threads::thread_id_repr_type id_;
            void* q_;
            hook_type slist_hook_;
        };

        typedef boost::intrusive::member_hook<
            queue_entry, queue_entry::hook_type,
            &queue_entry::slist_hook_
        > slist_option_type;

        typedef boost::intrusive::slist<
            queue_entry, slist_option_type,
            boost::intrusive::cache_last<true>,
            boost::intrusive::constant_time_size<true>
        > queue_type;

        struct reset_queue_entry
        {
            reset_queue_entry(queue_entry& e, queue_type& q)
              : e_(e), last_(q.last())
            {}

            ~reset_queue_entry()
            {
                if (e_.id_ != threads::invalid_thread_id_repr)
                {
                    queue_type* q = static_cast<queue_type*>(e_.q_);
                    q->erase(last_);     // remove entry from queue
                }
            }

            queue_entry& e_;
            queue_type::const_iterator last_;
        };

    public:
        HPX_EXPORT condition_variable();

        HPX_EXPORT ~condition_variable();

        HPX_EXPORT bool empty(
            boost::unique_lock<mutex_type> const& lock) const;

        HPX_EXPORT std::size_t size(
            boost::unique_lock<mutex_type> const& lock) const;

        // Return false if no more threads are waiting (returns true if queue
        // is non-empty).
        HPX_EXPORT bool notify_one(
            boost::unique_lock<mutex_type> lock, error_code& ec = throws);

        HPX_EXPORT void notify_all(
            boost::unique_lock<mutex_type> lock, error_code& ec = throws);

        HPX_EXPORT void abort_all(
            boost::unique_lock<mutex_type> lock);

        HPX_EXPORT threads::thread_state_ex_enum wait(
            boost::unique_lock<mutex_type>&& lock,
            char const* description, error_code& ec = throws);

        threads::thread_state_ex_enum wait(
            boost::unique_lock<mutex_type>& lock,
            char const* description, error_code& ec = throws)
        {
            relock_guard rl(lock);
            return wait(std::move(lock), description, ec);
        }

        threads::thread_state_ex_enum wait(
            boost::unique_lock<mutex_type>&& lock,
            error_code& ec = throws)
        {
            return wait(std::move(lock), "condition_variable::wait", ec);
        }

        threads::thread_state_ex_enum wait(
            boost::unique_lock<mutex_type>& lock,
            error_code& ec = throws)
        {
            return wait(lock, "condition_variable::wait", ec);
        }

        HPX_EXPORT threads::thread_state_ex_enum wait_until(
            boost::unique_lock<mutex_type>&& lock,
            util::steady_time_point const& abs_time,
            char const* description, error_code& ec = throws);

        threads::thread_state_ex_enum wait_until(
            boost::unique_lock<mutex_type>& lock,
            util::steady_time_point const& abs_time,
            char const* description, error_code& ec = throws)
        {
            relock_guard rl(lock);
            return wait_until(std::move(lock), abs_time, description, ec);
        }

        threads::thread_state_ex_enum wait_until(
            boost::unique_lock<mutex_type>&& lock,
            util::steady_time_point const& abs_time,
            error_code& ec = throws)
        {
            return wait_until(std::move(lock), abs_time,
                "condition_variable::wait_until", ec);
        }

        threads::thread_state_ex_enum wait_until(
            boost::unique_lock<mutex_type>& lock,
            util::steady_time_point const& abs_time,
            error_code& ec = throws)
        {
            return wait_until(lock, abs_time,
                "condition_variable::wait_until", ec);
        }

        threads::thread_state_ex_enum wait_for(
            boost::unique_lock<mutex_type>&& lock,
            util::steady_duration const& rel_time,
            char const* description, error_code& ec = throws)
        {
            return wait_until(std::move(lock), rel_time.from_now(), description, ec);
        }

        threads::thread_state_ex_enum wait_for(
            boost::unique_lock<mutex_type>& lock,
            util::steady_duration const& rel_time,
            char const* description, error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(), description, ec);
        }

        threads::thread_state_ex_enum wait_for(
            boost::unique_lock<mutex_type>&& lock,
            util::steady_duration const& rel_time,
            error_code& ec = throws)
        {
            return wait_until(std::move(lock), rel_time.from_now(),
                "condition_variable::wait_for", ec);
        }

        threads::thread_state_ex_enum wait_for(
            boost::unique_lock<mutex_type>& lock,
            util::steady_duration const& rel_time,
            error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(),
                "condition_variable::wait_for", ec);
        }

    private:
        template <typename Mutex>
        void abort_all(boost::unique_lock<Mutex> lock);

        // re-add the remaining items to the original queue
        HPX_EXPORT void prepend_entries(
            boost::unique_lock<mutex_type>& lock, queue_type& queue);

    private:
        queue_type queue_;
    };

}}}}

#endif
