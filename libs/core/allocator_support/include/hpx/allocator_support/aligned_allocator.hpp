//  Copyright (c) 2020 Thomas Heller
//  Copyright (c) 2023 Hartmut Kaiser
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
// this is currently used only for jemalloc and if a special API prefix is
// used for its APIs
#include <jemalloc/jemalloc.h>

namespace hpx::util::detail {

    [[nodiscard]] inline void* __aligned_alloc(
        std::size_t alignment, std::size_t size) noexcept
    {
        return HPX_PP_CAT(HPX_HAVE_JEMALLOC_PREFIX, aligned_alloc)(
            alignment, size);
    }

    inline void __aligned_free(void* p, std::size_t) noexcept
    {
        return HPX_PP_CAT(HPX_HAVE_JEMALLOC_PREFIX, free)(p);
    }
}    // namespace hpx::util::detail

#elif defined(HPX_HAVE_CXX17_STD_ALIGNED_ALLOC)

#include <cstdlib>

namespace hpx::util::detail {

    [[nodiscard]] inline void* __aligned_alloc(
        std::size_t alignment, std::size_t size) noexcept
    {
        return std::aligned_alloc(alignment, size);
    }

    inline void __aligned_free(void* p, std::size_t) noexcept
    {
        std::free(p);
    }
}    // namespace hpx::util::detail

#elif defined(HPX_HAVE_C11_ALIGNED_ALLOC)

#include <stdlib.h>

namespace hpx::util::detail {

    [[nodiscard]] inline void* __aligned_alloc(
        std::size_t alignment, std::size_t size) noexcept
    {
        return aligned_alloc(alignment, size);
    }

    inline void __aligned_free(void* p, std::size_t) noexcept
    {
        free(p);
    }
}    // namespace hpx::util::detail

#else    // !HPX_HAVE_CXX17_STD_ALIGNED_ALLOC && !HPX_HAVE_C11_ALIGNED_ALLOC

#include <cstdlib>

namespace hpx::util::detail {

    // provide our own (simple) implementation of aligned_alloc
    [[nodiscard]] inline void* __aligned_alloc(
        std::size_t alignment, std::size_t size) noexcept
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

        auto aligned_mem = static_cast<void*>(
            static_cast<char*>(allocated_mem) + sizeof(void*));

        std::align(alignment, size, aligned_mem, space);
        *(static_cast<void**>(aligned_mem) - 1) = allocated_mem;    //-V206

        return aligned_mem;
    }

    inline void __aligned_free(void* p, std::size_t) noexcept
    {
        if (nullptr != p)
        {
            std::free(*(static_cast<void**>(p) - 1));    //-V206
        }
    }
}    // namespace hpx::util::detail

#endif

namespace hpx::util::detail {

    template <typename Allocator>
    [[nodiscard]] void* __aligned_alloc(Allocator const& alloc,
        std::size_t alignment, std::size_t size) noexcept
    {
        using value_type =
            typename std::allocator_traits<Allocator>::value_type;
        using char_alloc = typename std::allocator_traits<
            Allocator>::template rebind_alloc<char>;

        std::size_t const s = size * sizeof(value_type);
        std::size_t space = s + alignment - 1;

        char_alloc a(alloc);
        void* allocated_mem = a.allocate(space + sizeof(void*));
        if (allocated_mem == nullptr)
        {
            return nullptr;
        }

        auto aligned_mem = static_cast<void*>(
            static_cast<char*>(allocated_mem) + sizeof(void*));

        std::align(alignment, size, aligned_mem, space);
        *(static_cast<void**>(aligned_mem) - 1) = allocated_mem;    //-V206

        return aligned_mem;
    }

    template <typename T>
    [[nodiscard]] void* __aligned_alloc(std::allocator<T> const&,
        std::size_t alignment, std::size_t size) noexcept
    {
        return __aligned_alloc(alignment, size);
    }

    template <typename Allocator>
    void __aligned_free(
        Allocator const& alloc, void* p, std::size_t size) noexcept
    {
        if (nullptr != p)
        {
            using char_alloc = typename std::allocator_traits<
                Allocator>::template rebind_alloc<char>;

            char_alloc a(alloc);
            a.deallocate(*(static_cast<void**>(p) - 1), size);    //-V206
        }
    }

    template <typename T>
    inline void __aligned_free(
        std::allocator<T> const&, void* p, std::size_t size) noexcept
    {
        __aligned_free(p, size);
    }
}    // namespace hpx::util::detail

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T = int, typename Allocator = std::allocator<T>>
    struct aligned_allocator
    {
    private:
        HPX_NO_UNIQUE_ADDRESS Allocator alloc;

    public:
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
            using other = aligned_allocator<U>;
        };

        using is_always_equal = std::true_type;
        using propagate_on_container_move_assignment = std::true_type;

        explicit aligned_allocator(
            Allocator const& alloc = Allocator{}) noexcept(noexcept(std::
                is_nothrow_copy_constructible_v<Allocator>))
          : alloc(alloc)
        {
        }

        template <typename U>
        explicit aligned_allocator(aligned_allocator<U> const& rhs) noexcept(
            noexcept(std::is_nothrow_copy_constructible_v<Allocator>))
          : alloc(rhs.alloc)
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

        [[nodiscard]] pointer allocate(size_type n, void const* = nullptr)
        {
            if (max_size() < n)
            {
                throw std::bad_array_new_length();
            }

            auto p = reinterpret_cast<pointer>(
                detail::__aligned_alloc(alloc, alignof(T), n * sizeof(T)));

            if (p == nullptr)
            {
                throw std::bad_alloc();
            }

            return p;
        }

        void deallocate(pointer p, size_type size) noexcept
        {
            detail::__aligned_free(alloc, p, size);
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
        aligned_allocator<T> const&, aligned_allocator<T> const&) noexcept
    {
        return true;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator!=(
        aligned_allocator<T> const&, aligned_allocator<T> const&) noexcept
    {
        return false;
    }
}    // namespace hpx::util

#include <hpx/config/warnings_suffix.hpp>
