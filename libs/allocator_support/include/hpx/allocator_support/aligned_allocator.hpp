//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include <hpx/preprocessor/cat.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T = int>
    struct aligned_allocator
    {
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef T const& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        template <typename U>
        struct rebind
        {
            typedef aligned_allocator<U> other;
        };

        typedef std::true_type is_always_equal;
        typedef std::true_type propagate_on_container_move_assignment;

        aligned_allocator() = default;

        template <typename U>
        explicit aligned_allocator(aligned_allocator<U> const&)
        {
        }

        pointer address(reference x) const noexcept
        {
            return &x;
        }

        const_pointer address(const_reference x) const noexcept
        {
            return &x;
        }

        pointer allocate(size_type n, void const* hint = nullptr)
        {
            return reinterpret_cast<pointer>(
                aligned_alloc(alignof(T), n * sizeof(T)));
        }

        void deallocate(pointer p, size_type n)
        {
            free(p);
        }

        size_type max_size() const noexcept
        {
            return (std::numeric_limits<size_type>::max)() / sizeof(T);
        }

        template <typename U, typename... Args>
        void construct(U* p, Args&&... args)
        {
            ::new ((void*) p) U(std::forward<Args>(args)...);
        }

        template <typename U>
        void destroy(U* p)
        {
            p->~U();
        }
    };

    template <typename T>
    constexpr bool operator==(
        aligned_allocator<T> const&, aligned_allocator<T> const&)
    {
        return true;
    }

    template <typename T>
    constexpr bool operator!=(
        aligned_allocator<T> const&, aligned_allocator<T> const&)
    {
        return false;
    }
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>
