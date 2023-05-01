//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/queue.hpp>

#include <cstddef>

namespace hpx::lockfree {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Alloc = std::allocator<T>>
    class caching_freelist : public lockfree::detail::freelist_stack<T, Alloc>
    {
        using base_type = lockfree::detail::freelist_stack<T, Alloc>;

    public:
        explicit caching_freelist(std::size_t n = 0)
          : lockfree::detail::freelist_stack<T, Alloc>(Alloc(), n)
        {
        }

        T* allocate()
        {
            return this->base_type::template allocate<true, false>();
        }

        void deallocate(T* n) noexcept
        {
            this->base_type::template deallocate<true>(n);
        }
    };

    template <typename T, typename Alloc = std::allocator<T>>
    class static_freelist : public lockfree::detail::freelist_stack<T, Alloc>
    {
        using base_type = lockfree::detail::freelist_stack<T, Alloc>;

    public:
        explicit static_freelist(std::size_t n = 0)
          : lockfree::detail::freelist_stack<T, Alloc>(Alloc(), n)
        {
        }

        T* allocate()
        {
            return this->base_type::template allocate<true, true>();
        }

        void deallocate(T* n) noexcept
        {
            this->base_type::template deallocate<true>(n);
        }
    };

    struct caching_freelist_t
    {
    };

    struct static_freelist_t
    {
    };
}    // namespace hpx::lockfree
