////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2012 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
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
    /// boost::mutex-compatible spinlock class
    struct spinlock
    {
    private:
#if defined(__ANDROID__) && defined(ANDROID)
        int v_;
#else
        boost::uint64_t v_;
#endif

        HPX_MOVABLE_BUT_NOT_COPYABLE(spinlock)

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
                        "spinlock::yield");
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
                    hpx::this_thread::suspend(hpx::threads::pending,
                        "local::spinlock::yield");
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
        spinlock(char const* const desc = "hpx::lcos::local::spinlock")
          : v_(0)
        {
            HPX_ITT_SYNC_CREATE(this, desc, "");
        }

        spinlock(spinlock && rhs)
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

        spinlock& operator=(spinlock && rhs)
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

            bool r = acquire_lock();

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
#if defined(BOOST_WINDOWS)
            boost::uint64_t r = BOOST_INTERLOCKED_EXCHANGE(&v_, 1);
            BOOST_COMPILER_FENCE
#else
            boost::uint64_t r = __sync_lock_test_and_set(&v_, 1);
#endif
            return r == 0;
        }

        void relinquish_lock()
        {
#if defined(BOOST_WINDOWS)
            BOOST_COMPILER_FENCE
            *const_cast<boost::uint64_t volatile*>(&v_) = 0;
#else
            __sync_lock_release(&v_);
#endif
        }

    public:
        typedef boost::unique_lock<spinlock> scoped_lock;
        typedef boost::detail::try_lock_wrapper<spinlock> scoped_try_lock;
    };
}}}

#endif // HPX_B3A83B49_92E0_4150_A551_488F9F5E1113

