//  Copyright (c) 2022-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/futures/future.hpp>

namespace hpx {

    template <typename T>
    struct future_or_value
    {
        constexpr future_or_value(T const& value)
          : data(value)
        {
        }

        constexpr future_or_value(T&& value) noexcept
          : data(HPX_MOVE(value))
        {
        }

        constexpr future_or_value(hpx::future<T>&& value) noexcept
          : data(HPX_MOVE(value))
        {
        }

        [[nodiscard]] constexpr bool has_value() const noexcept
        {
            return hpx::holds_alternative<T>(data);
        }
        [[nodiscard]] constexpr bool has_future() const noexcept
        {
            return hpx::holds_alternative<hpx::future<T>>(data);
        }

        constexpr T& get_value() &
        {
            return hpx::get<T>(data);
        }
        [[nodiscard]] constexpr T const& get_value() const&
        {
            return hpx::get<T>(data);
        }
        constexpr T get_value() &&
        {
            return hpx::get<T>(HPX_MOVE(data));
        }

        constexpr hpx::future<T>& get_future() &
        {
            return hpx::get<hpx::future<T>>(data);
        }
        [[nodiscard]] constexpr hpx::future<T> const& get_future() const&
        {
            return hpx::get<hpx::future<T>>(data);
        }
        constexpr hpx::future<T> get_future() &&
        {
            return hpx::get<hpx::future<T>>(HPX_MOVE(data));
        }

    private:
        hpx::variant<T, hpx::future<T>> data;
    };
}    // namespace hpx
