//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX20_STD_BIT_CAST)

#include <bit>

namespace hpx {

    using std::bit_cast;
}    // namespace hpx

#else

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace hpx {

    template <typename To, typename From>
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    std::enable_if_t<sizeof(To) == sizeof(From) &&
            std::is_trivially_copyable_v<From> &&
            std::is_trivially_copyable_v<To>,
        To> constexpr bit_cast(From const& src) noexcept
    {
        static_assert(std::is_trivially_constructible_v<To>,
            "This implementation additionally requires "
            "destination type to be trivially constructible");

        // NOLINTNEXTLINE(bugprone-sizeof-expression)
        constexpr std::size_t size = sizeof(To);

        To dst{};
        if constexpr (size == 1)
        {
            *(std::uint8_t*) &dst = *(std::uint8_t const*) &src;
        }
        else if constexpr (size == 2)
        {
            *(std::uint16_t*) &dst = *(std::uint16_t const*) &src;
        }
        else if constexpr (size == 4)
        {
            *(std::uint32_t*) &dst = *(std::uint32_t const*) &src;
        }
        else if constexpr (size == 8)
        {
            *(std::uint64_t*) &dst = *(std::uint64_t const*) &src;
        }
        else
        {
            std::memcpy(&dst, &src, sizeof(To));
        }
        return dst;
    }
}    // namespace hpx

#endif
