//  Copyright (c) 2018-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/type_support/construct_at.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#if defined(HPX_HAVE_JEMALLOC_PREFIX)
// this is currently used only for jemalloc and if a special API prefix is used
// for its APIs
#include <jemalloc/jemalloc.h>
#endif

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::util {

#if defined(HPX_HAVE_JEMALLOC_PREFIX)
    ///////////////////////////////////////////////////////////////////////////
    template <typename T = int>
    struct internal_allocator
    {
        using value_type = T;
        using pointer = T*;
        using const_pointer = T const*;
        using reference = T&;
        using const_reference = T const&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        template <typename U>
        struct rebind
        {
            using other = internal_allocator<U>;
        };

        using is_always_equal = std::true_type;
        using propagate_on_container_move_assignment = std::true_type;

        internal_allocator() = default;

        template <typename U>
        constexpr explicit internal_allocator(
            internal_allocator<U> const&) noexcept
        {
        }

        [[nodiscard]] static pointer address(reference x) noexcept
        {
            return &x;
        }

        [[nodiscard]] static const_pointer address(const_reference x) noexcept
        {
            return &x;
        }

        [[nodiscard]] static pointer allocate(
            size_type n, void const* hint = nullptr)
        {
            if (max_size() < n)
            {
                throw std::bad_array_new_length();
            }

            auto p = reinterpret_cast<pointer>(
                HPX_PP_CAT(HPX_HAVE_JEMALLOC_PREFIX, malloc)(n * sizeof(T)));
            if (p == nullptr)
            {
                throw std::bad_alloc();
            }
            return p;
        }

        static void deallocate(pointer p, size_type n) noexcept
        {
            HPX_PP_CAT(HPX_HAVE_JEMALLOC_PREFIX, free)(p);
        }

        [[nodiscard]] static constexpr size_type max_size() noexcept
        {
            return (std::numeric_limits<size_type>::max)() / sizeof(T);
        }

        template <typename U, typename... Args>
        static void construct(U* p, Args&&... args)
        {
            hpx::construct_at(p, HPX_FORWARD(Args, args)...);
        }

        template <typename U>
        static void destroy(U* p) noexcept
        {
            std::destroy_at(p);
        }
    };

    template <typename T>
    [[nodiscard]] constexpr bool operator==(
        internal_allocator<T> const&, internal_allocator<T> const&) noexcept
    {
        return true;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator!=(
        internal_allocator<T> const&, internal_allocator<T> const&) noexcept
    {
        return false;
    }
#else
    // fall back to system allocator if no special internal allocator is needed
    template <typename T = int>
    using internal_allocator = std::allocator<T>;
#endif
}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>
