//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_EVENT_SEP_29_2012_1918AM)
#define HPX_LCOS_EVENT_SEP_29_2012_1918AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/scoped_unlock.hpp>
#include <hpx/util/stringstream.hpp>

#include <boost/atomic.hpp>
#include <boost/cstdint.hpp>
#include <hpx/assert.hpp>
#include <boost/intrusive/slist.hpp>

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    /// Event semaphores can be used for synchronizing multiple threads that
    /// need to wait for an event to occur. When the event occurs, all threads
    /// waiting for the event are woken up.
    namespace detail
    {
        template <typename Mutex = lcos::local::spinlock>
        class event
        {
        private:
            typedef Mutex mutex_type;

            // define data structures needed for intrusive slist container used for
            // the queues
            struct queue_entry
            {
                typedef boost::intrusive::slist_member_hook<
                    boost::intrusive::link_mode<boost::intrusive::normal_link>
                > hook_type;

                queue_entry(threads::thread_id_type const& id)
                  : id_(id)
                {}

                threads::thread_id_type id_;
                hook_type list_hook_;
            };

            typedef boost::intrusive::member_hook<
                queue_entry, typename queue_entry::hook_type,
                &queue_entry::list_hook_
            > list_option_type;

            typedef boost::intrusive::slist<
                queue_entry, list_option_type,
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
                    if (e_.id_)
                        q_.erase(last_);     // remove entry from queue
                }

                queue_entry& e_;
                queue_type& q_;
                typename queue_type::const_iterator last_;
            };

        public:
            /// \brief Construct a new event semaphore
            event()
              : event_(false)
            {}

            ~event()
            {
                typename mutex_type::scoped_lock l(mtx_);

                if (!queue_.empty())
                {
                    LERR_(fatal)
                        << "lcos::event::~event:"
                           " queue is not empty, aborting threads";

                    while (!queue_.empty())
                    {
                        threads::thread_id_type id = queue_.front().id_;
                        queue_.front().id_ = threads::invalid_thread_id;
                        queue_.pop_front();

                        // we know that the id is actually the pointer to the thread
                        LERR_(fatal)
                                << "lcos::event::~event:"
                                << " pending thread: "
                                << get_thread_state_name(threads::get_thread_state(id))
                                << "(" << id << "): " << threads::get_thread_description(id);

                        // forcefully abort thread, do not throw
                        error_code ec(lightweight);
                        threads::set_thread_state(id, threads::pending,
                            threads::wait_abort, threads::thread_priority_default, ec);
                        if (ec)
                        {
                            LERR_(fatal)
                                << "lcos::event::~event:"
                                << " could not abort thread: "
                                << get_thread_state_name(threads::get_thread_state(id))
                                << "(" << id << "): " << threads::get_thread_description(id);
                        }
                    }
                }
            }

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

                typename mutex_type::scoped_lock l(mtx_);
                wait_locked(l);
            }

            /// \brief Release all threads waiting on this semaphore.
            void set()
            {
                event_.store(true, boost::memory_order_release);

                typename mutex_type::scoped_lock l(mtx_);
                set_locked(l);
            }

            /// \brief Reset the event
            void reset()
            {
                event_.store(false, boost::memory_order_release);
            }

        private:
            void wait_locked(typename mutex_type::scoped_lock& l)
            {
                HPX_ASSERT(l.owns_lock());

                while (!event_.load(boost::memory_order_acquire))
                {
                    queue_entry e(threads::get_self_id());
                    queue_.push_back(e);

                    reset_queue_entry r(e, queue_);

                    {
                        util::scoped_unlock<typename mutex_type::scoped_lock> ul(l);
                        this_thread::suspend(threads::suspended,
                            "lcos::event::wait");
                    }
                }
            }

            void set_locked(typename mutex_type::scoped_lock& l)
            {
                HPX_ASSERT(l.owns_lock());

                // swap the list
                queue_type queue;
                queue.swap(queue_);
                l.unlock();

                // release the threads
                while (!queue.empty())
                {
                    threads::thread_id_type id = queue.front().id_;
                    if (HPX_UNLIKELY(!id))
                    {
                        HPX_THROW_EXCEPTION(null_thread_id,
                            "lcos::event::set_locked",
                            "NULL thread id encountered");
                    }
                    queue.front().id_ = threads::invalid_thread_id;
                    queue.pop_front();
                    threads::set_thread_lco_description(id);
                    threads::set_thread_state(id, threads::pending);
                }
            }

            mutex_type mtx_;      ///< This mutex protects the queue.
            queue_type queue_;

            boost::atomic<bool> event_;
        };
    }

    typedef detail::event<> event;
}}}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif

