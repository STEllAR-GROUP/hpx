//  Copyright (c) 2024      Jacob Tucker
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <cstdint>

namespace hpx::lockfree {
    struct HPX_LOCKFREE_DCAS_ALIGNMENT uint128_type
    {
        std::uint64_t left = 0;
        std::uint64_t right = 0;

        uint128_type() = default;

        constexpr uint128_type(std::size_t l, std::size_t r) noexcept
          : left(l)
          , right(r)
        {
        }

        uint128_type(uint128_type const&) = default;
        uint128_type(uint128_type&&) = default;
        uint128_type& operator=(uint128_type const&) = default;
        uint128_type& operator=(uint128_type&&) = default;

        ~uint128_type() = default;

        friend constexpr bool operator==(uint128_type const& lhs,    //-V835
            uint128_type const& rhs) noexcept                        //-V835
        {
            return lhs.left == rhs.left && lhs.right == rhs.right;
        }

        friend constexpr bool operator!=(uint128_type const& lhs,    //-V835
            uint128_type const& rhs) noexcept                        //-V835
        {
            return !(lhs == rhs);
        }
    };
}    // namespace hpx::lockfree
