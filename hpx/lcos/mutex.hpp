//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Part of this code has been adopted from code published under the BSL by:
//
//  (C) Copyright 2005-7 Anthony Williams 
//  (C) Copyright 2007 David Deakins 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_BARRIER_JUN_23_2008_0530PM)
#define HPX_LCOS_BARRIER_JUN_23_2008_0530PM

#include <hpx/hpx_fwd.hpp>

#include <boost/noncopyable.hpp>
#include <boost/lockfree/fifo.hpp>

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
namespace hpx { namespace lcos { namespace detail
{
    // A mutex can be used to synchronize the access to an arbitrary resource
    class mutex 
    {
    private:
        BOOST_STATIC_CONSTANT(boost::int32_t, lock_flag_bit = 31);
        BOOST_STATIC_CONSTANT(boost::int32_t, event_set_flag_bit = 30);
        BOOST_STATIC_CONSTANT(boost::int32_t, lock_flag_value = 1 << lock_flag_bit);
        BOOST_STATIC_CONSTANT(boost::int32_t, event_set_flag_value = 1 << event_set_flag_bit);

    public:
        mutex()
          : active_count_(0)
        {}

        ~mutex()
        {
            BOOST_ASSERT(queue_.empty());
        }

        bool try_lock()
        {
            return !boost::lockfree::interlocked_bit_test_and_set(
                &active_count_, lock_flag_bit);
        }

        void lock()
        {
            BOOST_VERIFY(timed_lock(::boost::detail::get_system_time_sentinel()));
        }

        bool timed_lock(::boost::system_time const& wait_until)
        {
            using namespace boost::lockfree;
            if (!interlocked_bit_test_and_set(&active_count_, lock_flag_bit))
            {
                return true;
            }

            boost::int32_t old_count = active_count_;
            for(;;) {
                boost::int32_t const new_count = (old_count & lock_flag_value) ? 
                    (old_count + 1) : (old_count | lock_flag_value);
                boost::int32_t const current = 
                    interlocked_compare_exchange(&active_count_, old_count, new_count);
                if (current == old_count)
                {
                    break;
                }
                old_count = current;
            }

            if (old_count & lock_flag_value)
            {
                bool lock_acquired = false;
                do {
                    // wait for lock to get available
                    threads::thread_self& self = threads::get_self();
                    threads::thread_id_type id = self.get_thread_id();
                    queue_.enqueue(id);

                    // timeout at the given time, if appropriate
                    if (!wait_until.is_pos_infinity()) 
                        threads::set_thread_state(id, wait_until);

                    if (threads::wait_signaled != self.yield(threads::suspended))
                    {
                        // if this timed out, just return false
                        interlocked_decrement(&active_count_);
                        return false;
                    }

                    old_count &= ~lock_flag_value;
                    old_count |= event_set_flag_value;
                    for(;;) {
                        boost::int32_t const new_count = 
                            ((old_count & lock_flag_value) ? 
                                old_count : 
                                ((old_count-1) | lock_flag_value)) & ~event_set_flag_value;
                        boost::int32_t const current = 
                            interlocked_compare_exchange(&active_count_, old_count, new_count);
                        if (current == old_count)
                        {
                            break;
                        }
                        old_count = current;
                    }
                    lock_acquired = !(old_count & lock_flag_value);

                } while (!lock_acquired);
            }
            return true;
        }

        template<typename Duration>
        bool timed_lock(Duration const& timeout)
        {
            return timed_lock(boost::get_system_time()+timeout);
        }

        bool timed_lock(boost::xtime const& timeout)
        {
            return timed_lock(boost::posix_time::ptime(timeout));
        }

        void unlock()
        {
            using namespace boost::lockfree;
            boost::int32_t const offset = lock_flag_value;
            boost::int32_t const old_count = 
                interlocked_exchange_add(&active_count_, lock_flag_value);
            if (!(old_count & event_set_flag_value) && (old_count > offset))
            {
                if (!interlocked_bit_test_and_set(&active_count_, event_set_flag_bit))
                {
                    threads::thread_id_type id = 0;
                    if (queue_.dequeue(&id)) 
                        threads::set_thread_state(id);
                }
            }
        }

        typedef boost::unique_lock<mutex> scoped_lock;
        typedef boost::detail::try_lock_wrapper<mutex> scoped_try_lock;

    private:
        boost::int32_t active_count_;
        boost::lockfree::fifo<threads::thread_id_type> queue_;
    };

}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    ///////////////////////////////////////////////////////////////////////////
    class mutex : boost::noncopyable, public detail::mutex
    {
    public:
        mutex()
        {}
        ~mutex()
        {}

        typedef boost::unique_lock<mutex> scoped_lock;
        typedef boost::detail::try_lock_wrapper<mutex> scoped_try_lock;
    };

    typedef mutex try_mutex;

    ///////////////////////////////////////////////////////////////////////////
    class timed_mutex : boost::noncopyable, public detail::mutex
    {
    public:
        timed_mutex()
        {}
        ~timed_mutex()
        {}

        typedef boost::unique_lock<timed_mutex> scoped_timed_lock;
        typedef boost::detail::try_lock_wrapper<timed_mutex> scoped_try_lock;
        typedef scoped_timed_lock scoped_lock;
    };

}}

#endif

