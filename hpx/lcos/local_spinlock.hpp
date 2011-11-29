////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Hartmut Kaiser
//  Copyright (c) 2008 Peter Dimov
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B3A83B49_92E0_4150_A551_488F9F5E1113)
#define HPX_B3A83B49_92E0_4150_A551_488F9F5E1113

#include <hpx/util/itt_notify.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <boost/noncopyable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/config.hpp>

#if defined(BOOST_WINDOWS)
#  include <boost/smart_ptr/detail/spinlock_w32.hpp>
#else
#  include <boost/smart_ptr/detail/spinlock_sync.hpp>
#endif

namespace hpx { namespace lcos
{
    /// boost::mutex-compatible spinlock class
    struct local_spinlock : boost::noncopyable
    {
    private:
        boost::uint64_t v_;

        ///////////////////////////////////////////////////////////////////////////
        static void yield(std::size_t k)
        {
            //if (k < 4)
            if (k < 32)
            {
            }
            //if(k < 16)
            if(k < 256)
            {
#if defined(BOOST_SMT_PAUSE)
                BOOST_SMT_PAUSE
#endif
            }
            //else if(k < 32)
            /*
            else if(k < 512)
            {
                threads::suspend();
            }
            */
            else
            {
                //threads::suspend(boost::posix_time::microseconds(10));
                threads::suspend();
            }
        }

    public:
        local_spinlock() : v_(0)
        {
            HPX_ITT_SYNC_CREATE(this, "lcos::local_spinlock", "");
        }

        ~local_spinlock()
        {
            HPX_ITT_SYNC_DESTROY(this);
        }

        void lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            for (std::size_t k = 0; !try_lock(); ++k)
            {
                local_spinlock::yield(k);
            }

            HPX_ITT_SYNC_ACQUIRED(this);
        }

        bool try_lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

#if defined(BOOST_WINDOWS)
            boost::uint64_t r = BOOST_INTERLOCKED_EXCHANGE(&v_, 1);
            BOOST_COMPILER_FENCE
#else
            boost::uint64_t r = __sync_lock_test_and_set(&v_, 1);
#endif

            if (r == 0) {
                HPX_ITT_SYNC_ACQUIRED(this);
                return true;
            }

            HPX_ITT_SYNC_CANCEL(this);
            return false;
        }

        void unlock()
        {
            HPX_ITT_SYNC_RELEASING(this);

#if defined(BOOST_WINDOWS)
            BOOST_COMPILER_FENCE
            *const_cast<boost::uint64_t volatile*>(&v_) = 0;
#else
            __sync_lock_release(&v_);
#endif

            HPX_ITT_SYNC_RELEASED(this);
        }

        typedef boost::unique_lock<local_spinlock> scoped_lock;
        typedef boost::detail::try_lock_wrapper<local_spinlock> scoped_try_lock;
    };
}}

#endif // HPX_B3A83B49_92E0_4150_A551_488F9F5E1113

