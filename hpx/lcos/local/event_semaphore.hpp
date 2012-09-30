//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_EVENT_SEMAPHORE_SEP_29_2012_1918AM)
#define HPX_LCOS_EVENT_SEMAPHORE_SEP_29_2012_1918AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/stringstream.hpp>

#include <boost/atomic.hpp>
#include <boost/cstdint.hpp>
#include <boost/assert.hpp>
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
        class event_semaphore
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

                queue_entry(threads::thread_id_type id)
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
            event_semaphore()
              : event_(false) {}

            ~event_semaphore()
            {
                typename mutex_type::scoped_lock l(mtx_);

                if (!queue_.empty())
                {
                    LERR_(fatal)
                        << "lcos::event_semaphore::~event_semaphore:"
                           " queue is not empty, aborting threads";

                    while (!queue_.empty())
                    {
                        threads::thread_id_type id = queue_.front().id_;
                        queue_.front().id_ = 0;
                        queue_.pop_front();

                        // we know that the id is actually the pointer to the thread
                        LERR_(fatal)
                                << "lcos::event_semaphore::~event_semaphore:"
                                << " pending thread: "
                                << get_thread_state_name(threads::get_thread_state(id))
                                << "(" << id << "): " << threads::get_thread_description(id);

                        // forcefully abort thread, do not throw
                        error_code ec(lightweight);
                        threads::set_thread_state(id, threads::pending,
                            threads::wait_abort, threads::thread_priority_normal, ec);
                        if (ec)
                        {
                            LERR_(fatal)
                                << "lcos::event_semaphore::~event_semaphore:"
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
                // REVIEW: Is this memory order correct? See the example in
                // quickstart.
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

        private:
            void wait_locked(typename mutex_type::scoped_lock& l)
            {
                BOOST_ASSERT(l.owns_lock());

                while (!event_.load(boost::memory_order_acquire))
                {
                    threads::thread_self& self = threads::get_self();
    
                    queue_entry e(self.get_thread_id());
                    queue_.push_back(e);
    
                    reset_queue_entry r(e, queue_);
    
                    {
                        util::unlock_the_lock<typename mutex_type::scoped_lock> ul(l);
                        this_thread::suspend(threads::suspended,
                            "lcos::event_semaphore::wait");
                    }
                }
            }

            void set_locked(typename mutex_type::scoped_lock& l)
            {
                BOOST_ASSERT(l.owns_lock());

#if BOOST_VERSION < 103600
                // slist::swap has a bug in Boost 1.35.0
                while (!queue_.empty())
                {
                    threads::thread_id_type id = queue_.front().id_;
                    if (HPX_UNLIKELY(!id))
                    {
                        HPX_THROW_EXCEPTION(null_thread_id,
                            "lcos::event_semaphore::set_locked",
                            "NULL thread id encountered");
                    }
                    queue_.front().id_ = 0;
                    queue_.pop_front();
                    threads::set_thread_lco_description(id);
                    threads::set_thread_state(id, threads::pending);
                }
#else
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
                            "lcos::event_semaphore::set_locked",
                            "NULL thread id encountered");
                    }
                    queue.front().id_ = 0;
                    queue.pop_front();
                    threads::set_thread_lco_description(id);
                    threads::set_thread_state(id, threads::pending);
                }
#endif
            }

            mutex_type mtx_; ///< This mutex protects the queue.
            queue_type queue_;

            boost::atomic<bool> event_;
        };
    }

    typedef detail::event_semaphore<> event_semaphore;
}}}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif

