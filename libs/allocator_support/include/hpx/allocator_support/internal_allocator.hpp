//  Copyright (c) 2018 Hartmut Kaiser
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

#if defined(HPX_HAVE_JEMALLOC_PREFIX)
// this is currently used only for jemalloc and if a special API prefix is
// used for its APIs
#include <jemalloc/jemalloc.h>
#endif

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util {
#if defined(HPX_HAVE_JEMALLOC_PREFIX)
    ///////////////////////////////////////////////////////////////////////////
    template <typename T = int>
    struct internal_allocator
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
            typedef internal_allocator<U> other;
        };

        typedef std::true_type is_always_equal;
        typedef std::true_type propagate_on_container_move_assignment;

        internal_allocator() = default;

        template <typename U>
        explicit internal_allocator(internal_allocator<U> const&)
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
                HPX_PP_CAT(HPX_HAVE_JEMALLOC_PREFIX, malloc)(n * sizeof(T)));
        }

        void deallocate(pointer p, size_type n)
        {
            HPX_PP_CAT(HPX_HAVE_JEMALLOC_PREFIX, free)(p);
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
        internal_allocator<T> const&, internal_allocator<T> const&)
    {
        return true;
    }

    template <typename T>
    constexpr bool operator!=(
        internal_allocator<T> const&, internal_allocator<T> const&)
    {
        return false;
    }
#else
    // fall back to system allocator if no special internal allocator is needed
    template <typename T = int>
    using internal_allocator = std::allocator<T>;
#endif
}}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>
