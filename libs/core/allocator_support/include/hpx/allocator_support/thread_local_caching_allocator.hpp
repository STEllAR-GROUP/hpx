//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/config/defines.hpp>

#include <cstddef>
#include <memory>
#include <new>
#include <stack>
#include <type_traits>
#include <utility>

namespace hpx::util {

#if defined(HPX_ALLOCATOR_SUPPORT_HAVE_CACHING) &&                             \
    !((defined(HPX_HAVE_CUDA) && defined(__CUDACC__)) ||                       \
        defined(HPX_HAVE_HIP))
    ///////////////////////////////////////////////////////////////////////////
    template <typename T = char, typename Allocator = std::allocator<T>>
    struct thread_local_caching_allocator
    {
        HPX_NO_UNIQUE_ADDRESS Allocator alloc;

        using traits = std::allocator_traits<Allocator>;

        using value_type = typename traits::value_type;
        using pointer = typename traits::pointer;
        using const_pointer = typename traits::const_pointer;
        using size_type = typename traits::size_type;
        using difference_type = typename traits::difference_type;

        template <typename U>
        struct rebind
        {
            using other = thread_local_caching_allocator<U,
                typename traits::template rebind_alloc<U>>;
        };

        using is_always_equal = typename traits::is_always_equal;
        using propagate_on_container_copy_assignment =
            typename traits::propagate_on_container_copy_assignment;
        using propagate_on_container_move_assignment =
            typename traits::propagate_on_container_move_assignment;
        using propagate_on_container_swap =
            typename traits::propagate_on_container_swap;

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
                clear_cache();
            }

            pointer allocate(size_type n)
            {
                pointer p;
                if (data.empty())
                {
                    p = traits::allocate(alloc, n);
                    if (p == nullptr)
                    {
                        throw std::bad_alloc();
                    }
                }
                else
                {
                    p = data.top().first;
                    data.pop();
                }

                ++allocated;
                return p;
            }

            void deallocate(pointer p, size_type n) noexcept
            {
                data.push(std::make_pair(p, n));
                if (++deallocated > 2 * (allocated + 16))
                {
                    clear_cache();
                    allocated = 0;
                    deallocated = 0;
                }
            }

        private:
            void clear_cache() noexcept
            {
                while (!data.empty())
                {
                    traits::deallocate(
                        alloc, data.top().first, data.top().second);
                    data.pop();
                }
            }

            HPX_NO_UNIQUE_ADDRESS Allocator alloc;
            std::stack<std::pair<T*, size_type>> data;
            std::size_t allocated = 0;
            std::size_t deallocated = 0;
        };

        allocated_cache& cache()
        {
            thread_local allocated_cache allocated_data(alloc);
            return allocated_data;
        }

    public:
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

        [[nodiscard]] static constexpr pointer address(value_type& x) noexcept
        {
            return &x;
        }

        [[nodiscard]] static constexpr const_pointer address(
            value_type const& x) noexcept
        {
            return &x;
        }

        [[nodiscard]] pointer allocate(size_type n, void const* = nullptr)
        {
            if (max_size() < n)
            {
                throw std::bad_array_new_length();
            }
            return cache().allocate(n);
        }

        void deallocate(pointer p, size_type n) noexcept
        {
            cache().deallocate(p, n);
        }

        [[nodiscard]] constexpr size_type max_size() noexcept
        {
            return traits::max_size(alloc);
        }

        template <typename U, typename... Args>
        void construct(U* p, Args&&... args)
        {
            traits::construct(alloc, p, HPX_FORWARD(Args, args)...);
        }

        template <typename U>
        void destroy(U* p) noexcept
        {
            traits::destroy(alloc, p);
        }

        [[nodiscard]] friend constexpr bool operator==(
            thread_local_caching_allocator const& lhs,
            thread_local_caching_allocator const& rhs) noexcept
        {
            return lhs.alloc == rhs.alloc;
        }

        [[nodiscard]] friend constexpr bool operator!=(
            thread_local_caching_allocator const& lhs,
            thread_local_caching_allocator const& rhs) noexcept
        {
            return !(lhs == rhs);
        }
    };
#else
    template <typename T = char, typename Allocator = std::allocator<T>>
    using thread_local_caching_allocator = Allocator;
#endif
}    // namespace hpx::util
