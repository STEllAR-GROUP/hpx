//  Copyright (c) 2023-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/config/defines.hpp>
#include <hpx/assert.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace hpx::util {

#if defined(HPX_ALLOCATOR_SUPPORT_HAVE_CACHING) &&                             \
    !((defined(HPX_HAVE_CUDA) && defined(__CUDACC__)) ||                       \
        defined(HPX_HAVE_HIP))

    namespace detail {

        HPX_CORE_EXPORT void init_allocator_cache(
            std::size_t, std::function<void()>&& clear_cache);
        HPX_CORE_EXPORT std::pair<void*, std::size_t> allocate_from_cache(
            std::size_t) noexcept;
        [[nodiscard]] HPX_CORE_EXPORT bool cache_empty(std::size_t) noexcept;
        HPX_CORE_EXPORT void return_to_cache(
            std::size_t, void* p, std::size_t n);

        // maximal number of caches [0...max)
        inline constexpr int max_number_of_caches = 16;

        ///////////////////////////////////////////////////////////////////////
        constexpr int next_power_of_two(std::int64_t n) noexcept
        {
            int i = 0;
            for (--n; n > 0; n >>= 1)
            {
                ++i;
            }
            return i;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T = char, typename Allocator = std::allocator<T>>
    struct thread_local_caching_allocator
    {
        HPX_NO_UNIQUE_ADDRESS Allocator alloc;

    private:
        using traits = std::allocator_traits<Allocator>;

    public:
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

        explicit thread_local_caching_allocator(
            Allocator const& alloc = Allocator{}) noexcept(noexcept(std::
                is_nothrow_copy_constructible_v<Allocator>))
          : alloc(alloc)
        {
            // Note: capturing the allocator will be ok only as long as it
            // doesn't have any state as this lambda will be possibly called
            // very late during destruction of the thread_local cache.
            static_assert(std::is_empty_v<Allocator>,
                "Please don't use allocators with state in conjunction with "
                "the thread_local_caching_allocator");

            constexpr std::size_t num_cache =
                detail::next_power_of_two(sizeof(T));

            static_assert(num_cache < detail::max_number_of_caches,
                "This allocator does not support allocating objects larger "
                "than 2^16 bytes");

            auto f = [=]() mutable {
                while (!detail::cache_empty(num_cache))
                {
                    auto [p, n] = detail::allocate_from_cache(num_cache);
                    if (p != nullptr)
                    {
                        traits::deallocate(const_cast<Allocator&>(alloc),
                            static_cast<char*>(p), n);
                    }
                }
            };

            detail::init_allocator_cache(num_cache, HPX_MOVE(f));
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
            constexpr std::size_t num_cache =
                detail::next_power_of_two(sizeof(T));
            std::size_t N = n * (1ull << num_cache);

            if (max_size() < N)
            {
                throw std::bad_array_new_length();
            }

            auto [p, _] = detail::allocate_from_cache(num_cache);
            if (p == nullptr)
            {
                p = traits::allocate(alloc, N);
                if (p == nullptr)
                {
                    throw std::bad_alloc();
                }
            }
            return static_cast<pointer>(p);
        }

        void deallocate(pointer p, size_type n)
        {
            constexpr std::size_t num_cache =
                detail::next_power_of_two(sizeof(T));
            detail::return_to_cache(num_cache, p, n * (1ull << num_cache));
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
