//  Copyright (c) 2023-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/allocator_support/config/defines.hpp>

#if defined(HPX_ALLOCATOR_SUPPORT_HAVE_CACHING) &&                             \
    !((defined(HPX_HAVE_CUDA) && defined(__CUDACC__)) ||                       \
        defined(HPX_HAVE_HIP))

#include <hpx/allocator_support/thread_local_caching_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/type_support/static_reinit_interface.hpp>

#include <cstddef>
#include <functional>
#include <stack>
#include <utility>

namespace hpx::util::detail {

    ///////////////////////////////////////////////////////////////////////////
    struct allocated_cache
    {
        explicit allocated_cache() noexcept = default;

        void init(std::function<void()>&& clear)
        {
            if (!clear_cache)    // initialize once
            {
                clear_cache = HPX_MOVE(clear);
                util::reinit_register(std::function<void()>(), clear_cache);
            }
        }

        allocated_cache(allocated_cache const&) = delete;
        allocated_cache(allocated_cache&&) = delete;
        allocated_cache& operator=(allocated_cache const&) = delete;
        allocated_cache& operator=(allocated_cache&&) = delete;

        ~allocated_cache()
        {
            if (clear_cache)
            {
                clear_cache();
            }
        }

        std::pair<void*, std::size_t> allocate() noexcept
        {
            std::pair<void*, std::size_t> p{nullptr, 0};
            if (!data.empty())
            {
                p = data.top();
                data.pop();

                ++allocated;
            }
            return p;
        }

        void deallocate(void* p, std::size_t n)
        {
            data.emplace(p, n);
            if (++deallocated > 2 * (allocated + 16))
            {
                if (clear_cache)
                {
                    clear_cache();
                }

                allocated = 0;
                deallocated = 0;
            }
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return data.empty();
        }

    private:
        std::stack<std::pair<void*, std::size_t>> data;
        std::size_t allocated = 0;
        std::size_t deallocated = 0;
        std::function<void()> clear_cache;
    };

    ///////////////////////////////////////////////////////////////////////////
    allocated_cache& cache(std::size_t n)
    {
        HPX_ASSERT(n < max_number_of_caches);

        thread_local allocated_cache allocated_data[max_number_of_caches];
        return allocated_data[n];
    }

    void init_allocator_cache(
        std::size_t n, std::function<void()>&& clear_cache)
    {
        cache(n).init(HPX_MOVE(clear_cache));
    }

    std::pair<void*, std::size_t> allocate_from_cache(std::size_t n) noexcept
    {
        return cache(n).allocate();
    }

    void return_to_cache(std::size_t n, void* p, std::size_t const size)
    {
        cache(n).deallocate(p, size);
    }

    bool cache_empty(std::size_t n) noexcept
    {
        return cache(n).empty();
    }
}    // namespace hpx::util::detail

#endif
