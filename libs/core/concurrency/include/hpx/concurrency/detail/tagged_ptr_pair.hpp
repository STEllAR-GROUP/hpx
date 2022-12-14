//  Copyright (C) 2008-2011 Tim Blechmann
//  Copyright (C) 2011      Bryce Lelbach
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  tagged pointer pair, for aba prevention (intended for use with 128bit
//  atomics)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>    // for std::size_t
#include <cstdint>

namespace hpx::lockfree {

    struct HPX_LOCKFREE_DCAS_ALIGNMENT uint128_type
    {
        std::uint64_t left;
        std::uint64_t right;

        friend constexpr bool operator==(
            uint128_type const& lhs, uint128_type const& rhs) noexcept
        {
            return (lhs.left == rhs.left) && (lhs.right == rhs.right);
        }

        friend constexpr bool operator!=(
            uint128_type const& lhs, uint128_type const& rhs) noexcept
        {
            return !(lhs == rhs);
        }
    };

    template <typename Left, typename Right>
    struct HPX_LOCKFREE_DCAS_ALIGNMENT tagged_ptr_pair
    {
        using compressed_ptr_pair_t = uint128_type;
        using compressed_ptr_t = std::uint64_t;
        using tag_t = std::uint16_t;

        union HPX_LOCKFREE_DCAS_ALIGNMENT cast_unit
        {
            compressed_ptr_pair_t value;
            tag_t tags[8];
        };

        static constexpr std::size_t left_tag_index = 3;
        static constexpr std::size_t right_tag_index = 7;
        static constexpr compressed_ptr_t ptr_mask = 0xffffffffffff;

        static Left* extract_left_ptr(compressed_ptr_pair_t const& i) noexcept
        {
            return reinterpret_cast<Left*>(i.left & ptr_mask);
        }

        static Right* extract_right_ptr(compressed_ptr_pair_t const& i) noexcept
        {
            return reinterpret_cast<Right*>(i.right & ptr_mask);
        }

        static tag_t extract_left_tag(compressed_ptr_pair_t const& i) noexcept
        {
            cast_unit cu;
            cu.value.left = i.left;
            cu.value.right = i.right;
            return cu.tags[left_tag_index];
        }

        static tag_t extract_right_tag(compressed_ptr_pair_t const& i) noexcept
        {
            cast_unit cu;
            cu.value.left = i.left;
            cu.value.right = i.right;
            return cu.tags[right_tag_index];
        }

        template <typename IntegralL, typename IntegralR>
        static void pack_ptr_pair(compressed_ptr_pair_t& pair, Left* lptr,
            Right* rptr, IntegralL ltag, IntegralR rtag) noexcept
        {
            cast_unit ret;
            ret.value.left = reinterpret_cast<compressed_ptr_t>(lptr);
            ret.value.right = reinterpret_cast<compressed_ptr_t>(rptr);
            ret.tags[left_tag_index] = static_cast<tag_t>(ltag);
            ret.tags[right_tag_index] = static_cast<tag_t>(rtag);
            pair = ret.value;
        }

        /** uninitialized constructor */
        tagged_ptr_pair() {}

        template <typename IntegralL>
        tagged_ptr_pair(Left* lptr, Right* rptr, IntegralL ltag) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, ltag, 0);
        }

        template <typename IntegralL, typename IntegralR>
        tagged_ptr_pair(
            Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }

        /** copy constructors */
        tagged_ptr_pair(tagged_ptr_pair const& p) = default;

        tagged_ptr_pair(Left* lptr, Right* rptr) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, 0, 0);
        }

        /** unsafe set operations */
        /* @{ */
        tagged_ptr_pair& operator=(tagged_ptr_pair const& p) = default;

        void set(Left* lptr, Right* rptr) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, 0, 0);
        }

        void reset(Left* lptr, Right* rptr) noexcept
        {
            set(lptr, rptr, 0, 0);
        }

        template <typename IntegralL>
        void set(Left* lptr, Right* rptr, IntegralL ltag) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, ltag, 0);
        }

        template <typename IntegralL, typename IntegralR>
        void set(
            Left* lptr, Right* rptr, IntegralL ltag, IntegralR rtag) noexcept
        {
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }

        template <typename IntegralL>
        void reset(Left* lptr, Right* rptr, IntegralL ltag) noexcept
        {
            set(lptr, rptr, ltag, 0);
        }

        template <typename IntegralL, typename IntegralR>
        void reset(
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
            return (lhs.pair_ == rhs.pair_);
        }

        friend constexpr bool operator!=(
            tagged_ptr_pair const& lhs, tagged_ptr_pair const& rhs) noexcept
        {
            return !(lhs == rhs);
        }
        /* @} */

        /** pointer access */
        /* @{ */
        Left* get_left_ptr() const noexcept
        {
            return extract_left_ptr(pair_);
        }

        Right* get_right_ptr() const noexcept
        {
            return extract_right_ptr(pair_);
        }

        void set_left_ptr(Left* lptr) noexcept
        {
            Right* rptr = get_right_ptr();
            tag_t ltag = get_left_tag();
            tag_t rtag = get_right_tag();
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }

        void set_right_ptr(Right* rptr) noexcept
        {
            Left* lptr = get_left_ptr();
            tag_t ltag = get_left_tag();
            tag_t rtag = get_right_tag();
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }
        /* @} */

        /** tag access */
        /* @{ */
        tag_t get_left_tag() const noexcept
        {
            return extract_left_tag(pair_);
        }

        tag_t get_right_tag() const noexcept
        {
            return extract_right_tag(pair_);
        }

        template <typename Integral>
        void set_left_tag(Integral ltag) noexcept
        {
            Left* lptr = get_left_ptr();
            Right* rptr = get_right_ptr();
            tag_t rtag = get_right_tag();
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }

        template <typename Integral>
        void set_right_tag(Integral rtag) noexcept
        {
            Left* lptr = get_left_ptr();
            Right* rptr = get_right_ptr();
            tag_t ltag = get_left_tag();
            pack_ptr_pair(pair_, lptr, rptr, ltag, rtag);
        }
        /* @} */

        /** smart pointer support  */
        /* @{ */
        explicit operator bool() const noexcept
        {
            return (get_left_ptr() != 0) && (get_right_ptr() != 0);
        }
        /* @} */

    private:
        compressed_ptr_pair_t pair_;
    };
}    // namespace hpx::lockfree
