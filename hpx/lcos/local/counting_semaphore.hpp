//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_COUNTING_SEMAPHORE_OCT_16_2008_1007AM)
#define HPX_LCOS_COUNTING_SEMAPHORE_OCT_16_2008_1007AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/scoped_unlock.hpp>
#include <hpx/util/stringstream.hpp>

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
    /// A semaphore is a protected variable (an entity storing a value) or
    /// abstract data type (an entity grouping several variables that may or
    /// may not be numerical) which constitutes the classic method for
    /// restricting access to shared resources, such as shared memory, in a
    /// multiprogramming environment. Semaphores exist in many variants, though
    /// usually the term refers to a counting semaphore, since a binary
    /// semaphore is better known as a mutex. A counting semaphore is a counter
    /// for a set of available resources, rather than a locked/unlocked flag of
    /// a single resource. It was invented by Edsger Dijkstra. Semaphores are
    /// the classic solution to preventing race conditions in the dining
    /// philosophers problem, although they do not prevent resource deadlocks.
    ///
    /// Counting semaphores can be used for synchronizing multiple threads as
    /// well: one thread waiting for several other threads to touch (signal)
    /// the semaphore, or several threads waiting for one other thread to touch
    /// this semaphore.
    namespace detail
    {
        template <typename Mutex = lcos::local::spinlock>
        class HPX_EXPORT counting_semaphore
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
            /// \brief Construct a new counting semaphore
            ///
            /// \param value    [in] The initial value of the internal semaphore
            ///                 lock count. Normally this value should be zero
            ///                 (which is the default), values greater than zero
            ///                 are equivalent to the same number of signals pre-
            ///                 set, and negative values are equivalent to the
            ///                 same number of waits pre-set.
            counting_semaphore(boost::int64_t value = 0)
              : value_(value) {}

            ~counting_semaphore()
            {
                typename mutex_type::scoped_lock l(mtx_);

                if (!queue_.empty())
                {
                    LERR_(fatal)
                        << "lcos::counting_semaphore::~counting_semaphore:"
                           " queue is not empty, aborting threads";

                    while (!queue_.empty())
                    {
                        threads::thread_id_type id = queue_.front().id_;
                        queue_.front().id_ = threads::invalid_thread_id;
                        queue_.pop_front();

                        // we know that the id is actually the pointer to the thread
                        LERR_(fatal)
                                << "lcos::counting_semaphore::~counting_semaphore:"
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
                                << "lcos::counting_semaphore::~counting_semaphore:"
                                << " could not abort thread: "
                                << get_thread_state_name(threads::get_thread_state(id))
                                << "(" << id << "): " << threads::get_thread_description(id);
                        }
                    }
                }
            }

            /// \brief Wait for the semaphore to be signaled
            ///
            /// \param count    [in] The value by which the internal lock count will
            ///                 be decremented. At the same time this is the minimum
            ///                 value of the lock count at which the thread is not
            ///                 yielded.
            void wait(boost::int64_t count = 1)
            {
                typename mutex_type::scoped_lock l(mtx_);
                wait_locked(count, l);
            }

            /// \brief Try to wait for the semaphore to be signaled
            ///
            /// \param count    [in] The value by which the internal lock count will
            ///                 be decremented. At the same time this is the minimum
            ///                 value of the lock count at which the thread is not
            ///                 yielded.
            ///
            /// \returns        The function returns true if the calling thread was
            ///                 able to acquire the requested amount of credits.
            ///                 The function returns false if not sufficient credits
            ///                 are available at this point in time.
            bool try_wait(boost::int64_t count = 1)
            {
                typename mutex_type::scoped_lock l(mtx_);
                if (!(value_ < count)) {
                    // enter wait_locked only if there are sufficient credits
                    // available
                    wait_locked(count, l);
                    return true;
                }
                return false;
            }

            /// \brief Signal the semaphore
            ///
            ///
            void signal(boost::int64_t count = 1)
            {
                typename mutex_type::scoped_lock l(mtx_);
                signal_locked(count, l);
            }

            boost::int64_t signal_all()
            {
                typename mutex_type::scoped_lock l(mtx_);
                boost::int64_t count = static_cast<boost::int64_t>(queue_.size());
                signal_locked(count, l);
                return count;
            }

            template <typename Lock>
            void wait_locked(boost::int64_t count, Lock& l)
            {
                while (value_ < count)
                {
                    // we need to get the self anew for each round as it might
                    // get executed in a different thread from the previous one
                    queue_entry e(threads::get_self_id());
                    queue_.push_back(e);

                    reset_queue_entry r(e, queue_);

                    {
                        util::scoped_unlock<Lock> ul(l);
                        this_thread::suspend(threads::suspended,
                            "lcos::counting_semaphore::wait");
                    }
                }

                value_ -= count;
            }

        private:
            template <typename Lock>
            void signal_locked(boost::int64_t count, Lock& l)
            {
                value_ += count;
                if (value_ >= 0)
                {
                    // release all threads, they will figure out between themselves
                    // which one gets released from wait above
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
                                "lcos::counting_semaphore::signal_locked",
                                "NULL thread id encountered");
                        }
                        queue.front().id_ = threads::invalid_thread_id;
                        queue.pop_front();
                        threads::set_thread_lco_description(id);
                        threads::set_thread_state(id, threads::pending);
                    }
                }
            }

        private:
            mutex_type mtx_;
            boost::int64_t value_;
            queue_type queue_;
        };
    }

    typedef detail::counting_semaphore<> counting_semaphore;
}}}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif

