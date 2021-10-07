//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include <hpx/preprocessor/cat.hpp>

#if defined(HPX_HAVE_JEMALLOC_PREFIX)
// this is currently used only for jemalloc and if a special API prefix is
// used for its APIs
#include <jemalloc/jemalloc.h>

inline void* __aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    return HPX_PP_CAT(HPX_HAVE_JEMALLOC_PREFIX, aligned_alloc)(alignment, size);
}

inline void __aligned_free(void* p) noexcept
{
    return HPX_PP_CAT(HPX_HAVE_JEMALLOC_PREFIX, free)(p);
}

#elif defined(HPX_HAVE_CXX17_STD_ALIGNED_ALLOC)

#include <cstdlib>

inline void* __aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    return std::aligned_alloc(alignment, size);
}

inline void __aligned_free(void* p) noexcept
{
    std::free(p);
}

#elif defined(HPX_HAVE_C11_ALIGNED_ALLOC)

#include <stdlib.h>

inline void* __aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    return aligned_alloc(alignment, size);
}

inline void __aligned_free(void* p) noexcept
{
    free(p);
}

#else    // !HPX_HAVE_CXX17_STD_ALIGNED_ALLOC && !HPX_HAVE_C11_ALIGNED_ALLOC

#include <cstdlib>

// provide our own (simple) implementation of aligned_alloc
inline void* __aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    if (alignment < alignof(void*))
    {
        alignment = alignof(void*);
    }

    std::size_t space = size + alignment - 1;
    void* allocated_mem = std::malloc(space + sizeof(void*));
    if (allocated_mem == nullptr)
    {
        return nullptr;
    }

    void* aligned_mem =
        static_cast<void*>(static_cast<char*>(allocated_mem) + sizeof(void*));

    std::align(alignment, size, aligned_mem, space);
    *(static_cast<void**>(aligned_mem) - 1) = allocated_mem;

    return aligned_mem;
}

inline void __aligned_free(void* p) noexcept
{
    if (nullptr != p)
    {
        std::free(*(static_cast<void**>(p) - 1));
    }
}

#endif

#include <hpx/local/config/warnings_prefix.hpp>

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

        HPX_NODISCARD pointer allocate(size_type n, void const* = nullptr)
        {
            if (max_size() < n)
            {
                throw std::bad_array_new_length();
            }

            pointer p = reinterpret_cast<pointer>(
                __aligned_alloc(alignof(T), n * sizeof(T)));

            if (p == nullptr)
            {
                throw std::bad_alloc();
            }

            return p;
        }

        void deallocate(pointer p, size_type)
        {
            __aligned_free(p);
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

#include <hpx/local/config/warnings_suffix.hpp>
