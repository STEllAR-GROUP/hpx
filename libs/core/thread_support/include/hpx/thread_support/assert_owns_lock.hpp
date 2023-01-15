//  Copyright (c) 2013 Agustin Berge
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/type_support/unused.hpp>

#include <type_traits>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util::detail {

    HPX_HAS_MEMBER_XXX_TRAIT_DEF(owns_lock)

    template <typename Lock>
    constexpr void assert_owns_lock(Lock const&, int) noexcept
    {
    }

    template <typename Lock>
    constexpr void assert_doesnt_own_lock(Lock const&, int) noexcept
    {
    }

    template <typename Lock>
    std::enable_if_t<has_owns_lock_v<Lock>> assert_owns_lock(
        [[maybe_unused]] Lock& l, long) noexcept
    {
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26110)
#endif

        HPX_ASSERT_LOCKED(l, l.owns_lock());

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
    }

    template <typename Lock>
    std::enable_if_t<has_owns_lock_v<Lock>> assert_doesnt_own_lock(
        [[maybe_unused]] Lock& l, long) noexcept
    {
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26110)
#endif

        HPX_ASSERT_LOCKED(l, !l.owns_lock());

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
    }
}    // namespace hpx::util::detail

#define HPX_ASSERT_OWNS_LOCK(l) ::hpx::util::detail::assert_owns_lock(l, 0L)

#define HPX_ASSERT_DOESNT_OWN_LOCK(l)                                          \
    ::hpx::util::detail::assert_doesnt_own_lock(l, 0L)
