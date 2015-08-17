//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ASSERT_OWNS_LOCK_DEC_4_2013_0130PM)
#define HPX_UTIL_ASSERT_OWNS_LOCK_DEC_4_2013_0130PM

#include <hpx/config.hpp>
#include <hpx/util/assert.hpp>

#include <boost/thread/locks.hpp>
#include <boost/utility/declval.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace detail
{
    template <typename Lock>
    void assert_owns_lock(Lock const&, int) {}

#if !defined(HPX_DISABLE_ASSERTS) && !defined(BOOST_DISABLE_ASSERTS) && !defined(NDEBUG)
    template <typename Mutex>
    void assert_owns_lock(boost::unique_lock<Mutex> const& l, long)
    {
        HPX_ASSERT(l.owns_lock());
    }

    template <typename Mutex>
    void assert_owns_lock(boost::shared_lock<Mutex> const& l, long)
    {
        HPX_ASSERT(l.owns_lock());
    }

    template <typename Mutex>
    void assert_owns_lock(boost::upgrade_lock<Mutex> const& l, long)
    {
        HPX_ASSERT(l.owns_lock());
    }

#   if !defined(BOOST_NO_CXX11_DECLTYPE_N3276) && !defined(BOOST_NO_SFINAE_EXPR)
    template <typename Lock>
    decltype(boost::declval<Lock>().owns_lock())
    assert_owns_lock(Lock const& l, long)
    {
        HPX_ASSERT(l.owns_lock());
        return true;
    }
#   endif

#else

#   if !defined(BOOST_NO_CXX11_DECLTYPE_N3276) && !defined(BOOST_NO_SFINAE_EXPR)
    template <typename Lock>
    decltype(boost::declval<Lock>().owns_lock())
    assert_owns_lock(Lock const&, long)
    {
        return true;
    }
#   endif

#endif
}}}

#define HPX_ASSERT_OWNS_LOCK(l) ::hpx::util::detail::assert_owns_lock(l, 0L)

#endif
