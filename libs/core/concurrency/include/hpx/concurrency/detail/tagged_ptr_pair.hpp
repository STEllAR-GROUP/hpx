//  Copyright (C) 2008-2011 Tim Blechmann
//  Copyright (C) 2011      Bryce Lelbach
//  Copyright (c) 2022-2023 Hartmut Kaiser
//  Copyright (c) 2024      Jacob Tucker
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  tagged pointer pair, for aba prevention (intended for use with 128bit
//  atomics)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/detail/uint128_atomic.hpp>
#include <hpx/type_support/bit_cast.hpp>

#include <cstddef>    // for std::size_t
#include <cstdint>

namespace hpx::lockfree {
    namespace detail {

        template <std::size_t Size>
        struct ptr_mask;    // intentionally left unimplemented

        template <>
        struct ptr_mask<4>
        {
            static constexpr std::uint32_t value = 0xffffffff;
        };

        template <>
        struct ptr_mask<8>
        {
            static constexpr std::uint64_t value = 0xffffffffffff;
        };
    }    // namespace detail

    template <typename Left, typename Right>
    struct HPX_LOCKFREE_DCAS_ALIGNMENT tagged_ptr_pair
    {
        using compressed_ptr_pair_t = uint128_type;
        // compressed_ptr_t must be of the same size as a pointer
        using compressed_ptr_t = std::size_t;
        using tag_t = std::uint16_t;

        struct HPX_LOCKFREE_DCAS_ALIGNMENT cast_unit
        {
            union
            {
                compressed_ptr_pair_t value;
                tag_t tags[8];
            };

            explicit constexpr cast_unit(compressed_ptr_pair_t i) noexcept
              : value(i)
            {
            }
            constexpr cast_unit(compressed_ptr_t l, compressed_ptr_t r,
                tag_t ltag, tag_t rtag) noexcept
              : value(l, r)
            {
                tags[left_tag_index] = ltag;
                tags[right_tag_index] = rtag;
            }
        };

        static constexpr std::size_t left_tag_index = 3;
        static constexpr std::size_t right_tag_index = 7;
        static constexpr compressed_ptr_t ptr_mask =
            detail::ptr_mask<sizeof(compressed_ptr_t)>::value;

        static constexpr Left* extract_left_ptr(
            compressed_ptr_pair_t i) noexcept
        {
            return hpx::bit_cast<Left*>(
                static_cast<compressed_ptr_t>(i.left & ptr_mask));
        }

        static constexpr Right* extract_right_ptr(
            compressed_ptr_pair_t i) noexcept
        {
            return hpx::bit_cast<Right*>(
                static_cast<compressed_ptr_t>(i.right & ptr_mask));
        }

        static constexpr tag_t extract_left_tag(
            compressed_ptr_pair_t i) noexcept
        {
            cast_unit cu(i);
            return cu.tags[left_tag_index];
        }

        static constexpr tag_t extract_right_tag(
            compressed_ptr_pair_t i) noexcept
        {
            cast_unit cu(i);
            return cu.tags[right_tag_index];
        }

        template <typename IntegralL, typename IntegralR>
        static constexpr void pack_ptr_pair(compressed_ptr_pair_t& pair,
            Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag) noexcept
        {
            cast_unit ret(hpx::bit_cast<compressed_ptr_t>(lptr),
                hpx::bit_cast<compressed_ptr_t>(rptr), static_cast<tag_t>(ltag),
                static_cast<tag_t>(rtag));
            pair = ret.value;
        }

        /** uninitialized constructor */
        constexpr tagged_ptr_pair() {}    //-V730 //-V832

        template <typename IntegralL>
        constexpr tagged_ptr_pair(
            Left* lptr, Right* rptr, IntegralL ltag) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, ltag, 0);
        }

        template <typename IntegralL, typename IntegralR>
        constexpr tagged_ptr_pair(
            Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }

        /** copy constructors */
        tagged_ptr_pair(tagged_ptr_pair const& p) = default;
        tagged_ptr_pair(tagged_ptr_pair&& p) = default;

        constexpr tagged_ptr_pair(Left* lptr, Right* rptr) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, 0, 0);
        }

        /** unsafe set operations */
        /* @{ */
        tagged_ptr_pair& operator=(tagged_ptr_pair const& p) = default;
        tagged_ptr_pair& operator=(tagged_ptr_pair&& p) = default;

        ~tagged_ptr_pair() = default;

        constexpr void set(Left* lptr, Right* rptr) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, 0, 0);
        }

        constexpr void reset(Left* lptr, Right* rptr) noexcept
        {
            set(lptr, rptr, 0, 0);
        }

        template <typename IntegralL>
        constexpr void set(Left* lptr, Right* rptr, IntegralL ltag) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, ltag, 0);
        }

        template <typename IntegralL, typename IntegralR>
        constexpr void set(
            Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }

        template <typename IntegralL>
        constexpr void reset(Left* lptr, Right* rptr, IntegralL ltag) noexcept
        {
            set(lptr, rptr, ltag, 0);
        }

        template <typename IntegralL, typename IntegralR>
        constexpr void reset(
            Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag) noexcept
        {
            set(lptr, rptr, ltag, rtag);
        }
        /* @} */

        /** comparing semantics */
        /* @{ */
        friend constexpr bool operator==(
            tagged_ptr_pair const& lhs, tagged_ptr_pair const& rhs) noexcept
        {
            return lhs.pair_ == rhs.pair_;
        }

        friend constexpr bool operator!=(
            tagged_ptr_pair const& lhs, tagged_ptr_pair const& rhs) noexcept
        {
            return !(lhs == rhs);
        }
        /* @} */

        /** pointer access */
        /* @{ */
        constexpr Left* get_left_ptr() const noexcept
        {
            return extract_left_ptr(pair_);
        }

        constexpr Right* get_right_ptr() const noexcept
        {
            return extract_right_ptr(pair_);
        }

        constexpr void set_left_ptr(Left* lptr) noexcept
        {
            Right* rptr = get_right_ptr();
            tag_t const ltag = get_left_tag();
            tag_t const rtag = get_right_tag();
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }

        constexpr void set_right_ptr(Right* rptr) noexcept
        {
            Left* lptr = get_left_ptr();
            tag_t const ltag = get_left_tag();
            tag_t const rtag = get_right_tag();
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }
        /* @} */

        /** tag access */
        /* @{ */
        constexpr tag_t get_left_tag() const noexcept
        {
            return extract_left_tag(pair_);
        }

        constexpr tag_t get_right_tag() const noexcept
        {
            return extract_right_tag(pair_);
        }

        template <typename Integral>
        constexpr void set_left_tag(Integral ltag) noexcept
        {
            Left* lptr = get_left_ptr();
            Right* rptr = get_right_ptr();
            tag_t rtag = get_right_tag();
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }

        template <typename Integral>
        constexpr void set_right_tag(Integral rtag) noexcept
        {
            Left* lptr = get_left_ptr();
            Right* rptr = get_right_ptr();
            tag_t ltag = get_left_tag();
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }
        /* @} */

        /** smart pointer support  */
        /* @{ */
        explicit constexpr operator bool() const noexcept
        {
            return get_left_ptr() != nullptr && get_right_ptr() != nullptr;
        }
        /* @} */

    private:
        compressed_ptr_pair_t pair_;
    };
}    // namespace hpx::lockfree
