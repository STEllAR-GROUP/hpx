//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Part of this code has been adopted from code published under the BSL by:
//
//  (C) Copyright 2006-7 Anthony Williams 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_RECURSIVE_MUTEX_AUG_03_2009_0459PM)
#define HPX_LCOS_RECURSIVE_MUTEX_AUG_03_2009_0459PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/mutex.hpp>

#include <boost/atomic.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread/locks.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename UnderlyingMutex>
    struct recursive_mutex_impl
    {
        long recursion_count;
        boost::atomic<threads::thread_id_type> locking_thread_id;
        UnderlyingMutex mutex;

        recursive_mutex_impl(char const * const p)
          : recursion_count(0), locking_thread_id(0), mutex(p)
        {}

        bool try_lock()
        {
            threads::thread_id_type const current_thread_id = 
                threads::get_self().get_thread_id();

            return try_recursive_lock(current_thread_id) || 
                   try_basic_lock(current_thread_id);
        }

        void lock()
        {
            threads::thread_id_type const current_thread_id = 
                threads::get_self().get_thread_id();

            if (!try_recursive_lock(current_thread_id))
            {
                mutex.lock();
                locking_thread_id.exchange(current_thread_id);
                recursion_count = 1;
            }
        }
        bool timed_lock(::boost::system_time const& target)
        {
            threads::thread_id_type const current_thread_id = 
                threads::get_self().get_thread_id();

            return try_recursive_lock(current_thread_id) || 
                   try_timed_lock(current_thread_id, target);
        }
        template<typename Duration>
        bool timed_lock(Duration const& timeout)
        {
            return timed_lock(boost::get_system_time() + timeout);
        }

        void unlock()
        {
            if (!--recursion_count)
            {
                locking_thread_id.exchange((threads::thread_id_type)0);
                mutex.unlock();
            }
        }

    private:
        bool try_recursive_lock(threads::thread_id_type current_thread_id)
        {
            if(locking_thread_id.load(boost::memory_order_acquire) == 
                current_thread_id)
            {
                ++recursion_count;
                return true;
            }
            return false;
        }

        bool try_basic_lock(threads::thread_id_type current_thread_id)
        {
            if (mutex.try_lock())
            {
                locking_thread_id.exchange(current_thread_id);
                recursion_count = 1;
                return true;
            }
            return false;
        }

        bool try_timed_lock(threads::thread_id_type current_thread_id, 
            ::boost::system_time const& target)
        {
            if (mutex.timed_lock(target))
            {
                locking_thread_id.exchange(current_thread_id);
                recursion_count = 1;
                return true;
            }
            return false;
        }
    };

    typedef recursive_mutex_impl<detail::mutex> recursive_mutex;
    typedef recursive_mutex_impl<detail::mutex> recursive_timed_mutex;

}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos 
{
    ///////////////////////////////////////////////////////////////////////////
    class recursive_mutex
      : boost::noncopyable, public detail::recursive_mutex
    {
    public:
        recursive_mutex(char const * const p = "")
          : detail::recursive_mutex(p)
        {
        }
        ~recursive_mutex()
        {}

        typedef boost::unique_lock<recursive_mutex> scoped_lock;
        typedef boost::detail::try_lock_wrapper<recursive_mutex> scoped_try_lock;
    };

    typedef recursive_mutex recursive_try_mutex;

    class recursive_timed_mutex
      : boost::noncopyable, public detail::recursive_timed_mutex
    {
    public:
        recursive_timed_mutex(char const * const p = "")
          : detail::recursive_timed_mutex(p)
        {}
        ~recursive_timed_mutex()
        {}

        typedef boost::unique_lock<recursive_timed_mutex> scoped_timed_lock;
        typedef boost::detail::try_lock_wrapper<recursive_timed_mutex> scoped_try_lock;
        typedef scoped_timed_lock scoped_lock;
    };

}}

#endif
