//  Copyright (C) 2011, 2016 Tim Blechmann
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  boost lockfree: copy_payload helper

#pragma once

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx::lockfree::detail {

    template <typename T, typename U>
    HPX_FORCEINLINE constexpr void copy_payload(T& t, U& u)
    {
        if constexpr (std::is_convertible_v<T, U>)
        {
            u = t;
        }
        else
        {
            u = U(t);
        }
    }

    template <typename T>
    struct consume_via_copy
    {
        explicit constexpr consume_via_copy(T& out) noexcept
          : out_(out)
        {
        }

        template <typename U>
        constexpr void operator()(U& element) const
        {
            copy_payload(element, out_);
        }

        T& out_;
    };

    struct consume_noop
    {
        template <typename U>
        constexpr void operator()(U&&) const noexcept
        {
        }
    };
}    // namespace hpx::lockfree::detail
