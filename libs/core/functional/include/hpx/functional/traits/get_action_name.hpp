//  Copyright (c) 2007-2025 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/debugging.hpp>
#include <hpx/modules/serialization.hpp>
#if defined(HPX_HAVE_ITTNOTIFY) && HPX_HAVE_ITTNOTIFY != 0 &&                  \
    !defined(HPX_HAVE_APEX)
#include <hpx/modules/itt_notify.hpp>
#endif

namespace hpx::actions::detail {

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)

#if !defined(HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION)
    HPX_CXX_EXPORT template <typename Action>
    [[nodiscard]] char const* get_action_name() noexcept;
#else
    HPX_CXX_EXPORT template <typename Action>
    [[nodiscard]] char const* get_action_name() noexcept
    {
        /// If you encounter this assert while compiling code, that means that
        /// you have a HPX_REGISTER_ACTION macro somewhere in a source file,
        /// but the header in which the action is defined misses a
        /// HPX_REGISTER_ACTION_DECLARATION
        static_assert(hpx::traits::needs_automatic_registration_v<Action>,
            "HPX_REGISTER_ACTION_DECLARATION missing");
        return util::debug::type_id<Action>();
    }
#endif

#else    // HPX_HAVE_NETWORKING
    HPX_CXX_EXPORT template <typename Action>
    char const* get_action_name() noexcept
    {
        return util::debug::type_id<Action>();
    }
#endif

    ////////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_ITTNOTIFY) && HPX_HAVE_ITTNOTIFY != 0 &&                  \
    !defined(HPX_HAVE_APEX)

#if !defined(HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION)
    HPX_CXX_EXPORT template <typename Action>
    [[nodiscard]] util::itt::string_handle const&
    get_action_name_itt() noexcept;
#else
    HPX_CXX_EXPORT template <typename Action>
    [[nodiscard]] util::itt::string_handle const& get_action_name_itt() noexcept
    {
        static auto sh = util::itt::string_handle(get_action_name<Action>());
        return sh;
    }
#endif

#endif
}    // namespace hpx::actions::detail
