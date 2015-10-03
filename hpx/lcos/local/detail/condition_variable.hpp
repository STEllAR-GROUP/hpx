//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_DETAIL_CONDITION_VARIABLE_DEC_4_2013_0130PM)
#define HPX_LCOS_LOCAL_DETAIL_CONDITION_VARIABLE_DEC_4_2013_0130PM

#include <hpx/config.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/util/assert_owns_lock.hpp>
#include <hpx/util/unlock_guard.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <boost/intrusive/slist.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread/locks.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local { namespace detail
{
    class condition_variable : boost::noncopyable
    {
    private:
        // define data structures needed for intrusive slist container used for
        // the queues
        struct queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            queue_entry(threads::thread_id_repr_type const& id)
              : id_(id)
            {}

            threads::thread_id_repr_type id_;
            hook_type slist_hook_;
        };

        typedef boost::intrusive::member_hook<
            queue_entry, queue_entry::hook_type,
            &queue_entry::slist_hook_
        > slist_option_type;

        typedef boost::intrusive::slist<
            queue_entry, slist_option_type,
            boost::intrusive::cache_last<true>,
            boost::intrusive::constant_time_size<false>
        > queue_type;

        struct reset_queue_entry
        {
            reset_queue_entry(queue_entry& e, queue_type& q)
              : e_(e), q_(q), last_(q.last())
            {}

            ~reset_queue_entry()
            {
                if (e_.id_ != threads::invalid_thread_id_repr)
                    q_.erase(last_);     // remove entry from queue
            }

            queue_entry& e_;
            queue_type& q_;
            queue_type::const_iterator last_;
        };

    public:
        condition_variable()
        {}

        ~condition_variable()
        {
            if (!queue_.empty())
            {
                LERR_(fatal)
                    << "~condition_variable: queue is not empty, "
                       "aborting threads";

                local::no_mutex no_mtx;
                boost::unique_lock<local::no_mutex> lock(no_mtx);
                abort_all(std::move(lock));
            }
        }

        template <typename Mutex>
        bool empty(boost::unique_lock<Mutex> const& lock) const
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            return queue_.empty();
        }

        template <typename Mutex>
        std::size_t size(boost::unique_lock<Mutex> const& lock) const
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            return queue_.size();
        }

        // Return false if no more threads are waiting (returns true if queue
        // is non-empty).
        template <typename Mutex>
        bool notify_one(boost::unique_lock<Mutex> lock, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            if (!queue_.empty())
            {
                threads::thread_id_repr_type id = queue_.front().id_;

                // remove item from queue before error handling
                queue_.front().id_ = threads::invalid_thread_id_repr;
                queue_.pop_front();

                if (HPX_UNLIKELY(id == threads::invalid_thread_id_repr))
                {
                    lock.unlock();

                    HPX_THROWS_IF(ec, null_thread_id,
                        "condition_variable::notify_one",
                        "NULL thread id encountered");
                    return false;
                }

                bool not_empty = !queue_.empty();
                lock.unlock();

                threads::set_thread_state(threads::thread_id_type(
                    reinterpret_cast<threads::thread_data*>(id)),
                    threads::pending, threads::wait_signaled,
                    threads::thread_priority_default, ec);
                return not_empty;
            }

            if (&ec != &throws)
                ec = make_success_code();

            return false;
        }

        template <typename Mutex>
        void notify_all(boost::unique_lock<Mutex> lock, error_code& ec = throws)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            // swap the list
            queue_type queue;
            queue.swap(queue_);

            while (!queue.empty())
            {
                threads::thread_id_repr_type id = queue.front().id_;

                // remove item from queue before error handling
                queue.front().id_ = threads::invalid_thread_id_repr;
                queue.pop_front();

                if (HPX_UNLIKELY(id == threads::invalid_thread_id_repr))
                {
                    lock.unlock();

                    HPX_THROWS_IF(ec, null_thread_id,
                        "condition_variable::notify_all",
                        "NULL thread id encountered");
                    return;
                }

                // unlock while notifying thread as this can suspend
                util::unlock_guard<boost::unique_lock<Mutex> > unlock(lock);

                threads::set_thread_state(threads::thread_id_type(
                    reinterpret_cast<threads::thread_data*>(id)),
                    threads::pending, threads::wait_signaled,
                    threads::thread_priority_default, ec);
                if (ec) return;
            }

            if (&ec != &throws)
                ec = make_success_code();
        }

        template <typename Mutex>
        void abort_all(boost::unique_lock<Mutex> lock)
        {
            HPX_ASSERT_OWNS_LOCK(lock);

            // new threads might have been added while we were notifying
            while(!queue_.empty())
            {
                // swap the list
                queue_type queue;
                queue.swap(queue_);

                while (!queue.empty())
                {
                    threads::thread_id_repr_type id = queue.front().id_;

                    queue.front().id_ = threads::invalid_thread_id_repr;
                    queue.pop_front();

                    if (HPX_UNLIKELY(id == threads::invalid_thread_id_repr))
                    {
                        LERR_(fatal)
                            << "condition_variable::abort_all:"
                            << " NULL thread id encountered";
                        continue;
                    }

                    // we know that the id is actually the pointer to the thread
                    threads::thread_id_type tid(
                        reinterpret_cast<threads::thread_data*>(id));

                    LERR_(fatal)
                            << "condition_variable::abort_all:"
                            << " pending thread: "
                            << get_thread_state_name(
                                    threads::get_thread_state(tid))
                            << "(" << tid << "): "
                            << threads::get_thread_description(tid);

                    // unlock while notifying thread as this can suspend
                    util::unlock_guard<boost::unique_lock<Mutex> > unlock(lock);

                    // forcefully abort thread, do not throw
                    error_code ec(lightweight);
                    threads::set_thread_state(tid,
                        threads::pending, threads::wait_abort,
                        threads::thread_priority_default, ec);
                    if (ec)
                    {
                        LERR_(fatal)
                            << "condition_variable::abort_all:"
                            << " could not abort thread: "
                            << get_thread_state_name(
                                    threads::get_thread_state(tid))
                            << "(" << tid << "): "
                            << threads::get_thread_description(tid);
                    }
                }
            }
        }

        template <typename Mutex>
        threads::thread_state_ex_enum
        wait(boost::unique_lock<Mutex>& lock,
            char const* description, error_code& ec = throws)
        {
            HPX_ASSERT(threads::get_self_ptr() != 0);
            HPX_ASSERT_OWNS_LOCK(lock);

            // enqueue the request and block this thread
            queue_entry f(threads::get_self_id().get());
            queue_.push_back(f);

            reset_queue_entry r(f, queue_);
            threads::thread_state_ex_enum reason = threads::wait_unknown;
            {
                // yield this thread
                util::unlock_guard<boost::unique_lock<Mutex> > unlock(lock);
                reason = this_thread::suspend(threads::suspended, description, ec);
                if (ec) return threads::wait_unknown;
            }

            return (f.id_ != threads::invalid_thread_id_repr) ?
                threads::wait_timeout : reason;
        }

        template <typename Mutex>
        threads::thread_state_ex_enum
        wait(boost::unique_lock<Mutex>& lock, error_code& ec = throws)
        {
            return wait(lock, "condition_variable::wait", ec);
        }

        template <typename Mutex>
        threads::thread_state_ex_enum
        wait_until(boost::unique_lock<Mutex>& lock,
            util::steady_time_point const& abs_time,
            char const* description, error_code& ec = throws)
        {
            HPX_ASSERT(threads::get_self_ptr() != 0);
            HPX_ASSERT_OWNS_LOCK(lock);

            // enqueue the request and block this thread
            queue_entry f(threads::get_self_id().get());
            queue_.push_back(f);

            reset_queue_entry r(f, queue_);
            threads::thread_state_ex_enum reason = threads::wait_unknown;
            {
                // yield this thread
                util::unlock_guard<boost::unique_lock<Mutex> > unlock(lock);
                reason = this_thread::suspend(abs_time, description, ec);
                if (ec) return threads::wait_unknown;
            }

            return (f.id_ != threads::invalid_thread_id_repr) ?
                threads::wait_timeout : reason;
        }

        template <typename Mutex>
        threads::thread_state_ex_enum
        wait_until(boost::unique_lock<Mutex>& lock,
            util::steady_time_point const& abs_time,
            error_code& ec = throws)
        {
            return wait_until(lock, abs_time,
                "condition_variable::wait_until", ec);
        }

        template <typename Mutex>
        threads::thread_state_ex_enum
        wait_for(boost::unique_lock<Mutex>& lock,
            util::steady_duration const& rel_time,
            char const* description, error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(), description, ec);
        }

        template <typename Mutex>
        threads::thread_state_ex_enum
        wait_for(boost::unique_lock<Mutex>& lock,
            util::steady_duration const& rel_time,
            error_code& ec = throws)
        {
            return wait_until(lock, rel_time.from_now(),
                "condition_variable::wait_for", ec);
        }

    private:
        queue_type queue_;
    };

}}}}

#endif
