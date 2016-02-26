////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2012 Hartmut Kaiser
//  Copyright (c) 2008 Peter Dimov
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_LCOS_LOCAL_SPINLOCK_NO_BACKOFF)
#define HPX_LCOS_LOCAL_SPINLOCK_NO_BACKOFF

#include <hpx/config.hpp>
#include <hpx/config/emulate_deleted.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <boost/thread/locks.hpp>

#if defined(HPX_WINDOWS)
#  include <boost/smart_ptr/detail/spinlock.hpp>
#  if !defined( BOOST_SP_HAS_SYNC )
#    include <boost/detail/interlocked.hpp>
#  endif
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
    struct spinlock_no_backoff
    {
    private:
        HPX_NON_COPYABLE(spinlock_no_backoff);

    private:
        boost::uint64_t v_;

    public:
        spinlock_no_backoff() : v_(0)
        {
            HPX_ITT_SYNC_CREATE(this, "hpx::lcos::local::spinlock_no_backoff", "");
        }

        ~spinlock_no_backoff()
        {
            HPX_ITT_SYNC_DESTROY(this);
        }

        void lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            for (std::size_t k = 0; !try_lock(); ++k)
            {
            }

            HPX_ITT_SYNC_ACQUIRED(this);
            util::register_lock(this);
        }

        bool try_lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

#if !defined( BOOST_SP_HAS_SYNC )
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

#if !defined( BOOST_SP_HAS_SYNC )
            BOOST_COMPILER_FENCE
            *const_cast<boost::uint64_t volatile*>(&v_) = 0;
#else
            __sync_lock_release(&v_);
#endif

            HPX_ITT_SYNC_RELEASED(this);
            util::unregister_lock(this);
        }

        typedef boost::unique_lock<spinlock_no_backoff> scoped_lock;
        typedef boost::detail::try_lock_wrapper<spinlock_no_backoff> scoped_try_lock;
    };
}}}

#endif // HPX_B3A83B49_92E0_4150_A551_488F9F5E1113

