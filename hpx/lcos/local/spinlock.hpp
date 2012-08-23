////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2012 Hartmut Kaiser
//  Copyright (c) 2008 Peter Dimov
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B3A83B49_92E0_4150_A551_488F9F5E1113)
#define HPX_B3A83B49_92E0_4150_A551_488F9F5E1113

#include <hpx/util/move.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <boost/thread/locks.hpp>
#include <boost/config.hpp>

#if defined(BOOST_WINDOWS)
#  include <boost/smart_ptr/detail/spinlock_w32.hpp>
#else
#  include <boost/smart_ptr/detail/spinlock_sync.hpp>
#  if defined( __ia64__ ) && defined( __INTEL_COMPILER )
#    include <ia64intrin.h>
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    /// boost::mutex-compatible spinlock class
    struct spinlock
    {
    private:
        boost::uint64_t v_;

        BOOST_MOVABLE_BUT_NOT_COPYABLE(spinlock)

        ///////////////////////////////////////////////////////////////////////
        static void yield(std::size_t k)
        {
            if (k < 4)
            {
            }
            if(k < 16)
            {
#if defined(BOOST_SMT_PAUSE)
                BOOST_SMT_PAUSE
#endif
            }
            else if(k < 32 || k & 1)
            {
                if(hpx::threads::get_self_ptr())
                {
                    hpx::this_thread::suspend();
                }
                else
                {
#if defined(BOOST_WINDOWS)
                    Sleep(0);
#elif defined(BOOST_HAS_PTHREADS)
                    sched_yield();
#else
#endif
                }
            }
            else
            {
                if (hpx::threads::get_self_ptr())
                {
                    hpx::this_thread::suspend(boost::posix_time::microseconds(1));
                }
                else
                {
#if defined(BOOST_WINDOWS)
                    Sleep(1);
#elif defined(BOOST_HAS_PTHREADS)
                    // g++ -Wextra warns on {} or {0}
                    struct timespec rqtp = { 0, 0 };

                    // POSIX says that timespec has tv_sec and tv_nsec
                    // But it doesn't guarantee order or placement

                    rqtp.tv_sec = 0;
                    rqtp.tv_nsec = 1000;

                    nanosleep( &rqtp, 0 );
#else
#endif
                }
            }
        }

    public:
        spinlock() : v_(0)
        {
            HPX_ITT_SYNC_CREATE(this, "hpx::lcos::local::spinlock", "");
        }

        spinlock(BOOST_RV_REF(spinlock) rhs)
#if defined(BOOST_WINDOWS)
          : v_(BOOST_INTERLOCKED_EXCHANGE(&rhs.v_, 0))
#else
          : v_(__sync_lock_test_and_set(&rhs.v_, 0))
#endif
        {}

        ~spinlock()
        {
            HPX_ITT_SYNC_DESTROY(this);
        }

        spinlock& operator=(BOOST_RV_REF(spinlock) rhs)
        {
            if (this != &rhs) {
                unlock();
#if defined(BOOST_WINDOWS)
                v_ = BOOST_INTERLOCKED_EXCHANGE(&rhs.v_, 0);
#else
                v_ = __sync_lock_test_and_set(&rhs.v_, 0);
#endif
            }
            return *this;
        }

        void lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            for (std::size_t k = 0; !try_lock(); ++k)
            {
                spinlock::yield(k);
            }

            HPX_ITT_SYNC_ACQUIRED(this);
            util::register_lock(this);
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
                util::register_lock(this);
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
            util::unregister_lock(this);
        }

        typedef boost::unique_lock<spinlock> scoped_lock;
        typedef boost::detail::try_lock_wrapper<spinlock> scoped_try_lock;
    };
}}}

#endif // HPX_B3A83B49_92E0_4150_A551_488F9F5E1113

