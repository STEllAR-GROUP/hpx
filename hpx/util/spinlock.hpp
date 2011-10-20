////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_DF595582_FEBC_4EE0_A606_A1EEB171D770)
#define HPX_DF595582_FEBC_4EE0_A606_A1EEB171D770

#include <boost/noncopyable.hpp>
#include <boost/thread/locks.hpp>
#include <boost/smart_ptr/detail/spinlock.hpp>

namespace hpx { namespace util
{

// boost::mutex-compatible spinlock class
struct spinlock : boost::noncopyable
{
  private:
    boost::detail::spinlock m;

  public:
    spinlock()
    {
        boost::detail::spinlock l = BOOST_DETAIL_SPINLOCK_INIT;
        m = l;
    }

    void lock()
    { m.lock(); }

    bool try_lock()
    { return m.try_lock(); }

    void unlock()
    { m.unlock(); }

    typedef boost::detail::spinlock* native_handle_type;
    native_handle_type native_handle()
    { return &m; }

    typedef boost::unique_lock<spinlock> scoped_lock;
    typedef boost::detail::try_lock_wrapper<spinlock> scoped_try_lock;
};

}}

#endif // HPX_DF595582_FEBC_4EE0_A606_A1EEB171D770

