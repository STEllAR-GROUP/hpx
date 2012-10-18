//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Part of this code has been adopted from code published under the BSL by:
//
//  (C) Copyright 2005-7 Anthony Williams
//  (C) Copyright 2007 David Deakins
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_MUTEX_JUN_23_2008_0530PM)
#define HPX_LCOS_MUTEX_JUN_23_2008_0530PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/stringstream.hpp>

#include <boost/atomic.hpp>
#include <boost/noncopyable.hpp>
#include <boost/intrusive/slist.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <hpx/util/register_locks.hpp>

// Disable warning C4275: non dll-interface class used as base for dll-interface
// class
#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable: 4275)
#pragma warning(disable: 4251)
#endif

// Description of the mutex algorithm is explained here:
// http://lists.boost.org/Archives/boost/2006/09/110367.php
//
// The algorithm is:
//
// init():
//    active_count=0;
//    no semaphore
//
// lock():
//    atomic increment active_count
//    if new active_count ==1, that's us, so we've got the lock
//    else
//         get semaphore, and wait
//         now we've got the lock
//
// unlock():
//    atomic decrement active_count
//    if new active_count >0, then other threads are waiting,
//        so release semaphore.
//
// locked():
//    return active_count>0
//
// get_semaphore():
//    if there's already a semaphore associated with this mutex, return that
//    else
//        create new semaphore.
//        use atomic compare-and-swap to make this the associated semaphore if
//            none
//        if another thread beat us to it, and already set a semaphore, destroy
//            new one, and return already-set one
//        else return the new semaphore

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    template <typename T>
    inline bool interlocked_bit_test_and_set(boost::atomic<T>& x, long bit)
    {
        T const value = 1u << bit;
        boost::uint32_t old = x.load(boost::memory_order_acquire);
        do {
            boost::uint32_t tmp = old;
            if (x.compare_exchange_strong(tmp, T(old | value)))
                break;
            old = tmp;
        } while(true);
        return (old & value) != 0;
    }

    /// An exclusive-ownership mutex which implements Boost.Thread's
    /// TimedLockable concept.
    class HPX_EXPORT mutex : public boost::noncopyable
    {
    private:
        typedef lcos::local::spinlock mutex_type;

        BOOST_STATIC_CONSTANT(boost::uint32_t, lock_flag_bit = 31);
        BOOST_STATIC_CONSTANT(boost::uint32_t, lock_flag_value = 1u << lock_flag_bit);

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
            queue_entry, queue_entry::hook_type, &queue_entry::list_hook_
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
            queue_type::const_iterator last_;
        };

        ///////////////////////////////////////////////////////////////////////
        bool try_lock_internal()
        {
            return !interlocked_bit_test_and_set(active_count_, lock_flag_bit);
        }

        void mark_waiting_and_try_lock(boost::uint32_t& old_count)
        {
            for(;;)
            {
                boost::uint32_t new_count = (old_count & lock_flag_value) ?
                    (old_count + 1) : (old_count | lock_flag_value);

                boost::uint32_t tmp = old_count;
                if (active_count_.compare_exchange_strong(tmp, new_count))
                    break;
                old_count = tmp;
            }
        }

        void clear_waiting_and_try_lock(boost::uint32_t& old_count)
        {
            old_count &= ~lock_flag_value;
            for(;;)
            {
                boost::uint32_t new_count = (old_count & lock_flag_value) ?
                    old_count : ((old_count-1) | lock_flag_value);

                boost::uint32_t tmp = old_count;
                if (active_count_.compare_exchange_strong(tmp, new_count))
                    break;
                old_count = tmp;
            }
        }

        bool wait_for_single_object(
            ::boost::system_time const& wait_until = ::boost::system_time());

        void set_event();

    public:
        mutex(char const* const description = "")
          : active_count_(0), pending_events_(0), description_(description)
        {
            HPX_ITT_SYNC_CREATE(this, "lcos::mutex", description);
            HPX_ITT_SYNC_RENAME(this, "lcos::mutex");
        }

        ~mutex();

        /// Attempts to acquire ownership of the \a mutex. Never blocks.
        ///
        /// \returns \a true if ownership was acquired; otherwise, \a false.
        ///
        /// \throws Never throws.
        bool try_lock()
        {
            HPX_ITT_SYNC_PREPARE(this);
            bool got_lock = try_lock_internal();
            if (got_lock) {
                HPX_ITT_SYNC_ACQUIRED(this);
                util::register_lock(this);
            }
            else {
                HPX_ITT_SYNC_CANCEL(this);
            }
            return got_lock;
        }

        /// Acquires ownership of the \a mutex. Suspends the current
        /// HPX-thread if ownership cannot be obtained immediately.
        ///
        /// \throws Throws \a hpx#bad_parameter if an error occurs while
        ///         suspending. Throws \a hpx#yield_aborted if the mutex is
        ///         destroyed while suspended. Throws \a hpx#null_thread_id if
        ///         called outside of a HPX-thread.
        void lock()
        {
            HPX_ITT_SYNC_PREPARE(this);
            if (try_lock_internal()) {
                HPX_ITT_SYNC_ACQUIRED(this);
                util::register_lock(this);
                return;
            }

            boost::uint32_t old_count =
                active_count_.load(boost::memory_order_acquire);
            mark_waiting_and_try_lock(old_count);

            if (old_count & lock_flag_value)
            {
                // wait for lock to get available
                bool lock_acquired = false;
                do {
                    BOOST_VERIFY(!wait_for_single_object());
                    clear_waiting_and_try_lock(old_count);
                    lock_acquired = !(old_count & lock_flag_value);
                } while (!lock_acquired);
            }
            HPX_ITT_SYNC_ACQUIRED(this);
            util::register_lock(this);
        }

        /// Attempts to acquire ownership of the \a mutex. Suspends the
        /// current HPX-thread until \a wait_until if ownership cannot be obtained
        /// immediately.
        ///
        /// \returns \a true if ownership was acquired; otherwise, \a false.
        ///
        /// \throws Throws \a hpx#bad_parameter if an error occurs while
        ///         suspending. Throws \a hpx#yield_aborted if the mutex is
        ///         destroyed while suspended. Throws \a hpx#null_thread_id if
        ///         called outside of a HPX-thread.
        bool timed_lock(::boost::system_time const& wait_until);

        /// Attempts to acquire ownership of the \a mutex. Suspends the
        /// current HPX-thread until \a timeout if ownership cannot be obtained
        /// immediately.
        ///
        /// \returns \a true if ownership was acquired; otherwise, \a false.
        ///
        /// \throws Throws \a hpx#bad_parameter if an error occurs while
        ///         suspending. Throws \a hpx#yield_aborted if the mutex is
        ///         destroyed while suspended. Throws \a hpx#null_thread_id if
        ///         called outside of a HPX-thread.
        template<typename Duration>
        bool timed_lock(Duration const& timeout)
        {
            return timed_lock(boost::get_system_time()+timeout);
        }

        bool timed_lock(boost::xtime const& timeout)
        {
            return timed_lock(boost::posix_time::ptime(timeout));
        }

        /// Release ownership of the \a mutex.
        ///
        /// \throws Throws \a hpx#bad_parameter if an error occurs while
        ///         releasing the mutex.
        void unlock()
        {
            // We unregister ourselves before the actual unlock is executed as
            // the spinlock below might be there might be contention for the 
            // spinlock and the HPX-thread executing this unlock will be 
            // suspended.
            util::unregister_lock(this);

            HPX_ITT_SYNC_RELEASING(this);
            {
                mutex_type::scoped_lock l(mtx_);

                BOOST_ASSERT(active_count_ & lock_flag_value);
                active_count_ += lock_flag_value;
                set_event();
            }
            HPX_ITT_SYNC_RELEASED(this);
        }

        typedef boost::unique_lock<mutex> scoped_lock;
        typedef boost::detail::try_lock_wrapper<mutex> scoped_try_lock;

    private:
        mutable mutex_type mtx_;
        boost::atomic<boost::uint32_t> active_count_;
        queue_type queue_;
        boost::uint32_t pending_events_;
        char const* const description_;
    };
}}}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif

