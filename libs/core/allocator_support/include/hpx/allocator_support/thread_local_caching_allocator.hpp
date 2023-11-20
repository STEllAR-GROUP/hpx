//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <memory>
#include <new>
#include <stack>
#include <type_traits>
#include <utility>

namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
#if !((defined(HPX_HAVE_CUDA) && defined(__CUDACC__)) || defined(HPX_HAVE_HIP))
    template <typename T = char, typename Allocator = std::allocator<T>>
    struct thread_local_caching_allocator
    {
    private:
        struct allocated_cache
        {
            explicit allocated_cache(Allocator const& a) noexcept(
                noexcept(std::is_nothrow_copy_constructible_v<Allocator>))
              : alloc(a)
            {
            }

            allocated_cache(allocated_cache const&) = delete;
            allocated_cache(allocated_cache&&) = delete;
            allocated_cache& operator=(allocated_cache const&) = delete;
            allocated_cache& operator=(allocated_cache&&) = delete;

            ~allocated_cache()
            {
                using traits = std::allocator_traits<Allocator>;
                while (!data.empty())
                {
                    traits::deallocate(alloc, data.top(), 1);
                    data.pop();
                }
            }

            HPX_NO_UNIQUE_ADDRESS Allocator alloc;
            std::stack<T*> data;
        };

        std::stack<T*>& cache()
        {
            thread_local allocated_cache allocated_data(alloc);
            return allocated_data.data;
        }

    public:
        HPX_NO_UNIQUE_ADDRESS Allocator alloc;

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
            using other = thread_local_caching_allocator<U,
                typename std::allocator_traits<
                    Allocator>::template rebind_alloc<U>>;
        };

        using is_always_equal = std::true_type;
        using propagate_on_container_move_assignment = std::true_type;

        explicit thread_local_caching_allocator(
            Allocator const& alloc = Allocator{}) noexcept(noexcept(std::
                is_nothrow_copy_constructible_v<Allocator>))
          : alloc(alloc)
        {
        }

        template <typename U, typename Alloc>
        explicit thread_local_caching_allocator(
            thread_local_caching_allocator<U, Alloc> const&
                rhs) noexcept(noexcept(std::
                is_nothrow_copy_constructible_v<Alloc>))
          : alloc(rhs.alloc)
        {
        }

        [[nodiscard]] static constexpr pointer address(reference x) noexcept
        {
            return &x;
        }

        [[nodiscard]] static constexpr const_pointer address(
            const_reference x) noexcept
        {
            return &x;
        }

        [[nodiscard]] pointer allocate(size_type n, void const* = nullptr)
        {
            if (max_size() < n)
            {
                throw std::bad_array_new_length();
            }

            pointer p;

            if (auto& c = cache(); c.empty())
            {
                p = std::allocator_traits<Allocator>::allocate(alloc, n);
                if (p == nullptr)
                {
                    throw std::bad_alloc();
                }
            }
            else
            {
                p = c.top();
                c.pop();
            }

            return p;
        }

        void deallocate(pointer p, size_type) noexcept
        {
            cache().push(p);
        }

        [[nodiscard]] constexpr size_type max_size() noexcept
        {
            return std::allocator_traits<Allocator>::max_size(alloc);
        }

        template <typename U, typename... Args>
        void construct(U* p, Args&&... args)
        {
            std::allocator_traits<Allocator>::construct(
                alloc, p, HPX_FORWARD(Args, args)...);
        }

        template <typename U>
        void destroy(U* p) noexcept
        {
            std::allocator_traits<Allocator>::destroy(alloc, p);
        }
    };

    template <typename T>
    [[nodiscard]] constexpr bool operator==(
        thread_local_caching_allocator<T> const&,
        thread_local_caching_allocator<T> const&) noexcept
    {
        return true;
    }

    template <typename T>
    [[nodiscard]] constexpr bool operator!=(
        thread_local_caching_allocator<T> const&,
        thread_local_caching_allocator<T> const&) noexcept
    {
        return false;
    }
#else
    template <typename T = char, typename Allocator = std::allocator<T>>
    using thread_local_caching_allocator = Allocator;
#endif
}    // namespace hpx::util
