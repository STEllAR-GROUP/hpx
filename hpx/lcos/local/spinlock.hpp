////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2016 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  Copyright (c) 2008 Peter Dimov
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B3A83B49_92E0_4150_A551_488F9F5E1113)
#define HPX_B3A83B49_92E0_4150_A551_488F9F5E1113

#include <hpx/config.hpp>
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
#include <hpx/throw_exception.hpp>
#endif
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#if defined(HPX_WINDOWS)
#  include <boost/smart_ptr/detail/spinlock.hpp>
#  if !defined( BOOST_SP_HAS_SYNC )
#    include <boost/detail/interlocked.hpp>
#  endif
#else
#  if !defined(__ANDROID__) && !defined(ANDROID)
#    include <boost/smart_ptr/detail/spinlock_sync.hpp>
#    if defined( __ia64__ ) && defined( __INTEL_COMPILER )
#      include <ia64intrin.h>
#    endif
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
    HPX_API_EXPORT extern bool spinlock_break_on_deadlock;
    HPX_API_EXPORT extern std::size_t spinlock_deadlock_detection_limit;
#endif

    /// boost::mutex-compatible spinlock class
    struct spinlock
    {
    private:
        HPX_NON_COPYABLE(spinlock);

    private:
#if defined(__ANDROID__) && defined(ANDROID)
        int v_;
#else
        boost::uint64_t v_;
#endif

    public:
        ///////////////////////////////////////////////////////////////////////
        static void yield(std::size_t k)
        {
            if (k < 4) //-V112
            {
            }
#if defined(BOOST_SMT_PAUSE)
            else if(k < 16)
            {
                BOOST_SMT_PAUSE
            }
#endif
            else if(k < 32 || k & 1) //-V112
            {
                if (hpx::threads::get_self_ptr())
                {
                    hpx::this_thread::suspend(hpx::threads::pending,
                        "hpx::lcos::local::spinlock::yield");
                }
                else
                {
#if defined(HPX_WINDOWS)
                    Sleep(0);
#elif defined(BOOST_HAS_PTHREADS)
                    sched_yield();
#else
#endif
                }
            }
            else
            {
#ifdef HPX_HAVE_SPINLOCK_DEADLOCK_DETECTION
                if (spinlock_break_on_deadlock &&
                    k > spinlock_deadlock_detection_limit)
                {
                    HPX_THROW_EXCEPTION(deadlock,
                        "hpx::lcos::local::spinlock::yield",
                        "possible deadlock detected");
                }
#endif

                if (hpx::threads::get_self_ptr())
                {
                    hpx::this_thread::suspend(hpx::threads::pending,
                        "hpx::lcos::local::spinlock::yield");
                }
                else
                {
#if defined(HPX_WINDOWS)
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
        spinlock(char const* const desc = "hpx::lcos::local::spinlock")
          : v_(0)
        {
            HPX_ITT_SYNC_CREATE(this, desc, "");
        }

        ~spinlock()
        {
            HPX_ITT_SYNC_DESTROY(this);
        }

        void lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            for (std::size_t k = 0; !acquire_lock(); ++k)
            {
                spinlock::yield(k);
            }

            HPX_ITT_SYNC_ACQUIRED(this);
            util::register_lock(this);
        }

        bool try_lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            bool r = acquire_lock(); //-V707

            if (r) {
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

            relinquish_lock();

            HPX_ITT_SYNC_RELEASED(this);
            util::unregister_lock(this);
        }

    private:
        // returns whether the mutex has been acquired
        bool acquire_lock()
        {
#if !defined( BOOST_SP_HAS_SYNC )
            boost::uint64_t r = BOOST_INTERLOCKED_EXCHANGE(&v_, 1);
            BOOST_COMPILER_FENCE
#else
            boost::uint64_t r = __sync_lock_test_and_set(&v_, 1);
#endif
            return r == 0;
        }

        void relinquish_lock()
        {
#if !defined( BOOST_SP_HAS_SYNC )
            BOOST_COMPILER_FENCE
            *const_cast<boost::uint64_t volatile*>(&v_) = 0;
#else
            __sync_lock_release(&v_);
#endif
        }
    };
}}}

#endif // HPX_B3A83B49_92E0_4150_A551_488F9F5E1113

