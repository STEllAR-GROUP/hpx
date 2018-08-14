//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ASSERT_OWNS_LOCK_DEC_4_2013_0130PM)
#define HPX_UTIL_ASSERT_OWNS_LOCK_DEC_4_2013_0130PM

#include <hpx/config.hpp>
#include <hpx/traits/has_member_xxx.hpp>
#include <hpx/util/assert.hpp>

#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace detail
{
    HPX_HAS_MEMBER_XXX_TRAIT_DEF(owns_lock);

    template <typename Lock>
    void assert_owns_lock(Lock const&, int)
    {}

#if !defined(HPX_DISABLE_ASSERTS) && !defined(BOOST_DISABLE_ASSERTS) && !defined(NDEBUG)

    template <typename Lock>
    typename std::enable_if<
        has_owns_lock<Lock>::value
    >::type assert_owns_lock(Lock const& l, long)
    {
        HPX_ASSERT(l.owns_lock());
    }

#endif
}}}

#define HPX_ASSERT_OWNS_LOCK(l) ::hpx::util::detail::assert_owns_lock(l, 0L)

#endif
