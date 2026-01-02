////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2025 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>

// clang-format off
#if defined(HPX_HAVE_MORE_THAN_64_THREADS) ||                                  \
    (defined(HPX_HAVE_MAX_CPU_COUNT) && HPX_HAVE_MAX_CPU_COUNT > 64)

#  if defined(HPX_HAVE_MAX_CPU_COUNT)
#    include <bitset>
#  else
#    include <hpx/modules/datastructures.hpp>
#  endif

#endif
// clang-format on

namespace hpx::threads {

    /// \cond NOINTERNAL

    HPX_CXX_CORE_EXPORT [[nodiscard]] HPX_CORE_EXPORT unsigned int
    hardware_concurrency() noexcept;

#if !defined(HPX_HAVE_MORE_THAN_64_THREADS) ||                                 \
    (defined(HPX_HAVE_MAX_CPU_COUNT) && HPX_HAVE_MAX_CPU_COUNT <= 64)
    HPX_CXX_CORE_EXPORT using mask_type = std::uint64_t;
    HPX_CXX_CORE_EXPORT using mask_rvref_type = std::uint64_t;
    HPX_CXX_CORE_EXPORT using mask_cref_type = std::uint64_t;

    HPX_CXX_CORE_EXPORT constexpr std::uint64_t bits(std::size_t idx) noexcept
    {
        HPX_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        return static_cast<std::uint64_t>(1) << idx;
    }

    HPX_CXX_CORE_EXPORT constexpr bool any(mask_cref_type mask) noexcept
    {
        return mask != 0;
    }

    HPX_CXX_CORE_EXPORT constexpr mask_type not_(mask_cref_type mask) noexcept
    {
        return ~mask;
    }

    HPX_CXX_CORE_EXPORT constexpr bool test(
        mask_cref_type mask, std::size_t idx) noexcept
    {
        HPX_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        return (bits(idx) & mask) != 0;
    }

    HPX_CXX_CORE_EXPORT constexpr void set(
        mask_type& mask, std::size_t idx) noexcept
    {
        HPX_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        mask |= bits(idx);
    }

    HPX_CXX_CORE_EXPORT constexpr void unset(
        mask_type& mask, std::size_t idx) noexcept
    {
        HPX_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        mask &= not_(bits(idx));
    }

    HPX_CXX_CORE_EXPORT constexpr std::size_t mask_size(
        mask_cref_type /*mask*/) noexcept
    {
        return CHAR_BIT * sizeof(mask_type);
    }

    HPX_CXX_CORE_EXPORT constexpr void resize(
        mask_type& /*mask*/, [[maybe_unused]] std::size_t s) noexcept
    {
        HPX_ASSERT(s <= CHAR_BIT * sizeof(mask_type));
    }

    HPX_CXX_CORE_EXPORT constexpr std::size_t find_first(
        mask_cref_type mask) noexcept
    {
        if (mask)
        {
            std::size_t c = 0;    // Will count mask's trailing zero bits.

            // Set mask's trailing 0s to 1s and zero rest.
            mask = (mask ^ (mask - 1)) >> 1;
            for (/**/; mask; ++c)
                mask >>= 1;

            return c;
        }
        return ~static_cast<std::size_t>(0);
    }

    HPX_CXX_CORE_EXPORT constexpr bool equal(
        mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0) noexcept
    {
        return lhs == rhs;
    }

    // return true if at least one of the masks has a bit set
    HPX_CXX_CORE_EXPORT constexpr bool bit_or(
        mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0) noexcept
    {
        return (lhs | rhs) != 0;
    }

    // return true if at least one bit is set in both masks
    HPX_CXX_CORE_EXPORT constexpr bool bit_and(
        mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0) noexcept
    {
        return (lhs & rhs) != 0;
    }

    // returns the number of bits set, taken from:
    // https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
    HPX_CXX_CORE_EXPORT constexpr std::size_t count(mask_type mask) noexcept
    {
        std::size_t c = 0;    // c accumulates the total bits set in v
        for (; mask; ++c)
        {
            mask &= mask - 1;    // clear the least significant bit set
        }
        return c;
    }

    HPX_CXX_CORE_EXPORT constexpr void reset(mask_type& mask) noexcept
    {
        mask = 0ull;
    }

#else
#if defined(HPX_HAVE_MAX_CPU_COUNT)
    HPX_CXX_CORE_EXPORT using mask_type = std::bitset<HPX_HAVE_MAX_CPU_COUNT>;
    HPX_CXX_CORE_EXPORT using mask_rvref_type =
        std::bitset<HPX_HAVE_MAX_CPU_COUNT>&&;
    HPX_CXX_CORE_EXPORT using mask_cref_type =
        std::bitset<HPX_HAVE_MAX_CPU_COUNT> const&;

    HPX_CXX_CORE_EXPORT inline void set(
        mask_type& mask, std::size_t idx) noexcept;

#else
    HPX_CXX_CORE_EXPORT using mask_type =
        hpx::detail::dynamic_bitset<std::uint64_t>;
    HPX_CXX_CORE_EXPORT using mask_rvref_type =
        hpx::detail::dynamic_bitset<std::uint64_t>&&;
    HPX_CXX_CORE_EXPORT using mask_cref_type =
        hpx::detail::dynamic_bitset<std::uint64_t> const&;

    HPX_CXX_CORE_EXPORT inline void set(
        mask_type& mask, std::size_t idx) noexcept;
#endif

    HPX_CXX_CORE_EXPORT inline bool any(mask_cref_type mask) noexcept
    {
        return mask.any();
    }

    HPX_CXX_CORE_EXPORT inline mask_type not_(mask_cref_type mask)
    {
        return ~mask;
    }

    HPX_CXX_CORE_EXPORT inline bool test(
        mask_cref_type mask, std::size_t idx) noexcept
    {
        return mask.test(idx);
    }

    HPX_CXX_CORE_EXPORT inline void set(
        mask_type& mask, std::size_t idx) noexcept
    {
        mask.set(idx);
    }

    HPX_CXX_CORE_EXPORT inline void unset(
        mask_type& mask, std::size_t idx) noexcept
    {
        mask.set(idx, false);
    }

    HPX_CXX_CORE_EXPORT inline std::size_t mask_size(
        mask_cref_type mask) noexcept
    {
        return mask.size();
    }

    HPX_CXX_CORE_EXPORT inline void resize(
        [[maybe_unused]] mask_type& mask, [[maybe_unused]] std::size_t s)
    {
#if defined(HPX_HAVE_MAX_CPU_COUNT)
        HPX_ASSERT(s <= mask.size());
#else
        return mask.resize(s);
#endif
    }

    HPX_CXX_CORE_EXPORT inline std::size_t find_first(
        mask_cref_type mask) noexcept
    {
#if defined(HPX_HAVE_MAX_CPU_COUNT)
        if (mask.any())
        {
            for (std::size_t i = 0; i != HPX_HAVE_MAX_CPU_COUNT; ++i)
            {
                if (mask[i])
                    return i;
            }
        }
        return ~std::size_t(0);
#else
        return mask.find_first();
#endif
    }

    HPX_CXX_CORE_EXPORT inline bool equal(
        mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0) noexcept
    {
        return lhs == rhs;
    }

    // return true if at least one of the masks has a bit set
    HPX_CXX_CORE_EXPORT inline bool bit_or(
        mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return (lhs | rhs).any();
    }

    // return true if at least one bit is set in both masks
    HPX_CXX_CORE_EXPORT inline bool bit_and(
        mask_cref_type lhs, mask_cref_type rhs, std::size_t = 0)
    {
        return (lhs & rhs).any();
    }

    // returns the number of bits set
    HPX_CXX_CORE_EXPORT inline std::size_t count(mask_cref_type mask) noexcept
    {
        return mask.count();
    }

    HPX_CXX_CORE_EXPORT inline void reset(mask_type& mask) noexcept
    {
        mask.reset();
    }

#endif

    HPX_CXX_CORE_EXPORT HPX_CORE_EXPORT std::string to_string(mask_cref_type);
    /// \endcond
}    // namespace hpx::threads
