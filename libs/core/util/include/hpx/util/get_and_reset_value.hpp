//  Copyright (c) 2007-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <atomic>
#include <cstdint>
#include <vector>

namespace hpx::util {

    // helper function for counter evaluation
    HPX_CXX_EXPORT [[nodiscard]] constexpr std::uint64_t get_and_reset_value(
        std::uint64_t& value, bool const reset) noexcept
    {
        std::uint64_t const result = value;
        if (reset)
            value = 0;
        return result;
    }

    HPX_CXX_EXPORT [[nodiscard]] constexpr std::int64_t get_and_reset_value(
        std::int64_t& value, bool const reset) noexcept
    {
        std::int64_t const result = value;
        if (reset)
            value = 0;
        return result;
    }

    HPX_CXX_EXPORT template <typename T>
    [[nodiscard]] T get_and_reset_value(
        std::atomic<T>& value, bool const reset) noexcept
    {
        if (reset)
            return value.exchange(0, std::memory_order_acq_rel);
        return value.load(std::memory_order_relaxed);
    }

    HPX_CXX_EXPORT [[nodiscard]] inline std::vector<std::int64_t>
    get_and_reset_value(
        std::vector<std::int64_t>& value, bool const reset) noexcept
    {
        std::vector<std::int64_t> result = value;
        if (reset)
            value.clear();

        return result;
    }
}    // namespace hpx::util
