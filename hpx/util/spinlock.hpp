////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Hartmut Kaiser
//  Copyright (c) 2008 Peter Dimov
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_DF595582_FEBC_4EE0_A606_A1EEB171D770)
#define HPX_DF595582_FEBC_4EE0_A606_A1EEB171D770

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/itt_notify.hpp>

#include <boost/noncopyable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/smart_ptr/detail/spinlock.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace detail 
{
#if defined(BOOST_WINDOWS)
    ///////////////////////////////////////////////////////////////////////////
    inline void yield(unsigned k)
    {
        if(k < 4)
        {
        }
#if defined( BOOST_SMT_PAUSE )
        else if(k < 16)
        {
            BOOST_SMT_PAUSE
        }
#endif
        else if(k < 32)
        {
            threads::suspend();
        }
        else
        {
            threads::suspend(boost::posix_time::microseconds(100));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    class spinlock
    {
    public:
        long v_;

    public:
        bool try_lock()
        {
            long r = BOOST_INTERLOCKED_EXCHANGE(&v_, 1);

            BOOST_COMPILER_FENCE

            return r == 0;
        }

        void lock()
        {
            for (unsigned k = 0; !try_lock(); ++k)
            {
                yield(k);
            }
        }

        void unlock()
        {
            BOOST_COMPILER_FENCE
            *const_cast<long volatile*>(&v_) = 0;
        }

    public:
        class scoped_lock
        {
        private:
            spinlock & sp_;

            scoped_lock( scoped_lock const & );
            scoped_lock & operator=( scoped_lock const & );

        public:
            explicit scoped_lock( spinlock & sp ): sp_( sp )
            {
                sp.lock();
            }

            ~scoped_lock()
            {
                sp_.unlock();
            }
        };
    };
#else
    using boost::detail::spinlock;
#endif
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{

// boost::mutex-compatible spinlock class
struct spinlock : boost::noncopyable
{
  private:
    hpx::util::detail::spinlock m;

  public:
    spinlock()
    {
        HPX_ITT_SYNC_CREATE(this, "util::spinlock", "");

        hpx::util::detail::spinlock l = BOOST_DETAIL_SPINLOCK_INIT;
        m = l;
    }

    ~spinlock()
    {
        HPX_ITT_SYNC_DESTROY(this);
    }

    void lock()
    {
        HPX_ITT_SYNC_PREPARE(this);
        m.lock();
        HPX_ITT_SYNC_ACQUIRED(this);
    }

    bool try_lock()
    {
        HPX_ITT_SYNC_PREPARE(this);
        if (m.try_lock()) {
            HPX_ITT_SYNC_ACQUIRED(this);
            return true;
        }
        HPX_ITT_SYNC_CANCEL(this);
        return false;
    }

    void unlock()
    {
        HPX_ITT_SYNC_RELEASING(this);
        m.unlock();
        HPX_ITT_SYNC_RELEASED(this);
    }

    typedef hpx::util::detail::spinlock* native_handle_type;

    native_handle_type native_handle()
    {
        return &m;
    }

    typedef boost::unique_lock<spinlock> scoped_lock;
    typedef boost::detail::try_lock_wrapper<spinlock> scoped_try_lock;
};

}}

#endif // HPX_DF595582_FEBC_4EE0_A606_A1EEB171D770

