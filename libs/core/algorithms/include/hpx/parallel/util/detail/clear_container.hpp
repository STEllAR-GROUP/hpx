//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/futures/future.hpp>

#include <array>
#include <cstddef>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::util::detail {

    // make sure iterators embedded in function object that is attached to
    // futures are invalidated
    template <typename Cont>
    constexpr void clear_container(Cont&&) noexcept
    {
    }

    template <typename T>
    constexpr void clear_container(std::vector<hpx::future<T>>& v) noexcept
    {
        v.clear();
    }

    template <typename T>
    constexpr void clear_container(
        std::vector<hpx::shared_future<T>>& v) noexcept
    {
        v.clear();
    }

    template <typename T, std::size_t N>
    constexpr void clear_container(std::array<hpx::future<T>, N>& arr) noexcept
    {
        for (auto& f : arr)
        {
            f = hpx::future<T>();
        }
    }

    template <typename T, std::size_t N>
    constexpr void clear_container(
        std::array<hpx::shared_future<T>, N>& arr) noexcept
    {
        for (auto& f : arr)
        {
            f = hpx::shared_future<T>();
        }
    }
}    // namespace hpx::parallel::util::detail
