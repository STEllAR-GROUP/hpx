////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2018 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2008 Peter Dimov
//  Copyright (c) 2018 Patrick Diehl
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B3A83B49_92E0_4150_A551_488F9F5E1113)
#define HPX_B3A83B49_92E0_4150_A551_488F9F5E1113

#include <hpx/config.hpp>

#include <hpx/basic_execution/register_locks.hpp>
#include <hpx/concurrency/itt_notify.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/detail/yield_k.hpp>

#include <cstddef>
#include <cstdint>

#if defined(HPX_WINDOWS)
#  include <boost/smart_ptr/detail/spinlock.hpp>
#  if !defined(BOOST_SP_HAS_SYNC)
#    include <boost/detail/interlocked.hpp>
#  endif
#else
#  if !defined(__ANDROID__) && !defined(ANDROID)
#    include <boost/smart_ptr/detail/spinlock.hpp>
#    if defined(__ia64__) && defined(__INTEL_COMPILER)
#      include <ia64intrin.h>
#    endif
#  endif
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local
{
    // std::mutex-compatible spinlock class
    struct spinlock
    {
    public:
        HPX_NON_COPYABLE(spinlock);

    private:
#if defined(__ANDROID__) && defined(ANDROID)
        int v_;
#else
        std::uint64_t v_;
#endif

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
                util::detail::yield_k(k, "hpx::lcos::local::spinlock::lock",
                    hpx::threads::pending_boost);
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
#if !defined(BOOST_SP_HAS_SYNC)
            std::uint64_t r = BOOST_INTERLOCKED_EXCHANGE(&v_, 1);
            HPX_COMPILER_FENCE;
#else
            std::uint64_t r = __sync_lock_test_and_set(&v_, 1);
#endif
            return r == 0;
        }

        void relinquish_lock()
        {
#if !defined(BOOST_SP_HAS_SYNC)
            HPX_COMPILER_FENCE;
            *const_cast<std::uint64_t volatile*>(&v_) = 0;
#else
            __sync_lock_release(&v_);
#endif
        }
    };
}}}

#endif // HPX_B3A83B49_92E0_4150_A551_488F9F5E1113

