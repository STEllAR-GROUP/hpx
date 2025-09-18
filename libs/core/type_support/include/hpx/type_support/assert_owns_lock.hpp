//  Copyright (c) 2013 Agustin Berge
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/concepts.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util::detail {

    HPX_HAS_MEMBER_XXX_TRAIT_DEF(HPX_CORE_MODULE_EXPORT_EXTERN, owns_lock)

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename Lock>
    constexpr void assert_owns_lock(Lock const&, int) noexcept
        requires(!has_owns_lock_v<Lock>)
    {
    }

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename Lock>
    constexpr void assert_doesnt_own_lock(Lock const&, int) noexcept
        requires(!has_owns_lock_v<Lock>)
    {
    }

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename Lock>
    void assert_owns_lock([[maybe_unused]] Lock& l, long) noexcept
        requires(has_owns_lock_v<Lock>)
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

    HPX_CORE_MODULE_EXPORT_EXTERN template <typename Lock>
    void assert_doesnt_own_lock([[maybe_unused]] Lock& l, long) noexcept
        requires(has_owns_lock_v<Lock>)
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
