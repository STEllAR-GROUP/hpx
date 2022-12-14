//  Copyright (C) 2008, 2009, 2016 Tim Blechmann, based on code by Cory Nelson
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  tagged pointer, for aba prevention

#pragma once

#include <hpx/config.hpp>

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
        union cast_unit
        {
            compressed_ptr_t value;
            tag_t tag[4];
        };

        static constexpr int tag_index = 3;
        static constexpr compressed_ptr_t ptr_mask =
            0xffffffffffffUL;    // (1L<<48L)-1;

        static T* extract_ptr(compressed_ptr_t const& i) noexcept
        {
            return reinterpret_cast<T*>(i & ptr_mask);
        }

        static tag_t extract_tag(compressed_ptr_t const& i) noexcept
        {
            cast_unit cu;
            cu.value = i;
            return cu.tag[tag_index];
        }

        static compressed_ptr_t pack_ptr(T* ptr, tag_t tag) noexcept
        {
            cast_unit ret;
            ret.value = compressed_ptr_t(ptr);
            ret.tag[tag_index] = tag;
            return ret.value;
        }

    public:
        /** uninitialized constructor */
        constexpr tagged_ptr() noexcept    //: ptr(0), tag(0)
        {
        }

        /** copy constructor */
        tagged_ptr(tagged_ptr const& p) = default;

        explicit tagged_ptr(T* p, tag_t t = 0) noexcept
          : ptr(pack_ptr(p, t))
        {
        }

        /** unsafe set operation */
        /* @{ */
        tagged_ptr& operator=(tagged_ptr const& p) = default;

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
        T* get_ptr() const noexcept
        {
            return extract_ptr(ptr);
        }

        void set_ptr(T* p) noexcept
        {
            tag_t tag = get_tag();
            ptr = pack_ptr(p, tag);
        }
        /* @} */

        /** tag access */
        /* @{ */
        tag_t get_tag() const noexcept
        {
            return extract_tag(ptr);
        }

        tag_t get_next_tag() const noexcept
        {
            tag_t next = (get_tag() + 1u) & (std::numeric_limits<tag_t>::max)();
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
        T& operator*() const noexcept
        {
            return *get_ptr();
        }

        T* operator->() const noexcept
        {
            return get_ptr();
        }

        explicit operator bool() const noexcept
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
