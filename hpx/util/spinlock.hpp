////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_DF595582_FEBC_4EE0_A606_A1EEB171D770)
#define HPX_DF595582_FEBC_4EE0_A606_A1EEB171D770

#include <hpx/config.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/register_locks.hpp>

#include <boost/thread/locks.hpp>
#include <boost/smart_ptr/detail/spinlock.hpp>

namespace hpx { namespace util
{

/// boost::mutex-compatible spinlock class
struct spinlock
{
  private:
    HPX_NON_COPYABLE(spinlock);

  private:
    boost::detail::spinlock m;

  public:
    spinlock(char const* /*desc*/ = 0)
    {
        HPX_ITT_SYNC_CREATE(this, "util::spinlock", "");

        boost::detail::spinlock l = BOOST_DETAIL_SPINLOCK_INIT;
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
        util::register_lock(this);
    }

    bool try_lock()
    {
        HPX_ITT_SYNC_PREPARE(this);
        if (m.try_lock()) {
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
        m.unlock();
        HPX_ITT_SYNC_RELEASED(this);
        util::unregister_lock(this);
    }

    typedef boost::detail::spinlock* native_handle_type;

    native_handle_type native_handle()
    {
        return &m;
    }

    typedef boost::unique_lock<spinlock> scoped_lock;
    typedef boost::detail::try_lock_wrapper<spinlock> scoped_try_lock;
};

}}

#endif // HPX_DF595582_FEBC_4EE0_A606_A1EEB171D770

