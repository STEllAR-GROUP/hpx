//  Copyright (C) 2008, 2016 Tim Blechmann
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  tagged pointer, for aba prevention

#pragma once

#include <hpx/config.hpp>

#include <cstddef> /* for std::size_t */
#include <limits>

namespace hpx::lockfree::detail {

    template <typename T>
    class HPX_LOCKFREE_DCAS_ALIGNMENT tagged_ptr
    {
    public:
        using tag_t = std::size_t;
        using index_t = T*;

        /** uninitialized constructor */
        constexpr tagged_ptr() noexcept    //: ptr(0), tag(0)
        {
        }

        tagged_ptr(tagged_ptr const& p) = default;
        tagged_ptr(tagged_ptr&& p) = default;

        explicit constexpr tagged_ptr(T* p, tag_t t = 0) noexcept
          : ptr(p)
          , tag(t)
        {
        }

        /** unsafe set operation */
        /* @{ */
        tagged_ptr& operator=(tagged_ptr const& p) = default;
        tagged_ptr& operator=(tagged_ptr&& p) = default;

        ~tagged_ptr() = default;

        constexpr void set(T* p, tag_t t) noexcept
        {
            ptr = p;
            tag = t;
        }
        /* @} */

        /** comparing semantics */
        /* @{ */
        constexpr bool operator==(tagged_ptr const& rhs) const noexcept
        {
            return (ptr == rhs.ptr) && (tag == rhs.tag);
        }

        constexpr bool operator!=(tagged_ptr const& rhs) const noexcept
        {
            return !(*this == rhs);
        }
        /* @} */

        /** pointer access */
        /* @{ */
        constexpr T* get_ptr() const noexcept
        {
            return ptr;
        }

        constexpr void set_ptr(T* p) noexcept
        {
            ptr = p;
        }
        /* @} */

        /** tag access */
        /* @{ */
        constexpr tag_t get_tag() const noexcept
        {
            return tag;
        }

        constexpr tag_t get_next_tag() const noexcept
        {
            tag_t const next =
                (get_tag() + 1) & (std::numeric_limits<tag_t>::max)();
            return next;
        }

        constexpr void set_tag(tag_t t) noexcept
        {
            tag = t;
        }
        /* @} */

        /** smart pointer support  */
        /* @{ */
        constexpr T& operator*() const noexcept
        {
            return *ptr;
        }

        constexpr T* operator->() const noexcept
        {
            return ptr;
        }

        explicit constexpr operator bool() const noexcept
        {
            return ptr != nullptr;
        }
        /* @} */

    protected:
        T* ptr;
        tag_t tag;
    };

}    // namespace hpx::lockfree::detail
