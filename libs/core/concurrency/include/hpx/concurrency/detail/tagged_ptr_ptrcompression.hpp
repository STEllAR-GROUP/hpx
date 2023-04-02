//  Copyright (C) 2008, 2009, 2016 Tim Blechmann, based on code by Cory Nelson
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  tagged pointer, for aba prevention

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/bit_cast.hpp>

#include <cstdint>
#include <limits>

namespace hpx::lockfree::detail {

#if defined(HPX_LOCKFREE_PTR_COMPRESSION)

    template <typename T>
    class tagged_ptr
    {
        using compressed_ptr_t = std::uint64_t;

    public:
        using tag_t = std::uint16_t;
        using index_t = T*;

    private:
        struct cast_unit
        {
            union
            {
                compressed_ptr_t value;
                tag_t tag[4];
            };

            explicit constexpr cast_unit(compressed_ptr_t i) noexcept
              : value(i)
            {
            }

            constexpr cast_unit(compressed_ptr_t i, tag_t t) noexcept
              : value(i)
            {
                tag[tag_index] = t;
            }
        };

        static constexpr int tag_index = 3;
        static constexpr compressed_ptr_t ptr_mask =
            0xffffffffffffUL;    // (1L<<48L)-1;

        static constexpr T* extract_ptr(compressed_ptr_t i) noexcept
        {
            return hpx::bit_cast<T*>(i & ptr_mask);
        }

        static constexpr tag_t extract_tag(compressed_ptr_t i) noexcept
        {
            cast_unit cu(i);
            return cu.tag[tag_index];
        }

        static constexpr compressed_ptr_t pack_ptr(T* ptr, tag_t tag) noexcept
        {
            cast_unit ret(hpx::bit_cast<compressed_ptr_t>(ptr), tag);
            return ret.value;
        }

    public:
        /** uninitialized constructor */
        constexpr tagged_ptr() noexcept    //-V730 //-V832
        {
        }

        /** copy constructor */
        tagged_ptr(tagged_ptr const& p) = default;
        tagged_ptr(tagged_ptr&& p) = default;

        explicit constexpr tagged_ptr(T* p, tag_t t = 0) noexcept
          : ptr(pack_ptr(p, t))
        {
        }

        /** unsafe set operation */
        /* @{ */
        tagged_ptr& operator=(tagged_ptr const& p) = default;
        tagged_ptr& operator=(tagged_ptr&& p) = default;

        ~tagged_ptr() = default;

        void set(T* p, tag_t t) noexcept
        {
            ptr = pack_ptr(p, t);
        }
        /* @} */

        /** comparing semantics */
        /* @{ */
        friend constexpr bool operator==(
            tagged_ptr const& lhs, tagged_ptr const& rhs) noexcept
        {
            return (lhs.ptr == rhs.ptr);
        }

        friend constexpr bool operator!=(
            tagged_ptr const& lhs, tagged_ptr const& rhs) noexcept
        {
            return !(lhs == rhs);
        }
        /* @} */

        /** pointer access */
        /* @{ */
        constexpr T* get_ptr() const noexcept
        {
            return extract_ptr(ptr);
        }

        void set_ptr(T* p) noexcept
        {
            tag_t const tag = get_tag();
            ptr = pack_ptr(p, tag);
        }
        /* @} */

        /** tag access */
        /* @{ */
        constexpr tag_t get_tag() const noexcept
        {
            return extract_tag(ptr);
        }

        constexpr tag_t get_next_tag() const noexcept
        {
            tag_t const next =
                (get_tag() + 1u) & (std::numeric_limits<tag_t>::max)();
            return next;
        }

        void set_tag(tag_t t) noexcept
        {
            T* p = get_ptr();
            ptr = pack_ptr(p, t);
        }
        /* @} */

        /** smart pointer support  */
        /* @{ */
        constexpr T& operator*() const noexcept
        {
            return *get_ptr();
        }

        constexpr T* operator->() const noexcept
        {
            return get_ptr();
        }

        constexpr explicit operator bool() const noexcept
        {
            return get_ptr() != nullptr;
        }
        /* @} */

    protected:
        compressed_ptr_t ptr;
    };
#else
#error unsupported platform
#endif

}    // namespace hpx::lockfree::detail
